import os
import torch
import argparse
import numpy as np
from scipy.stats import hmean
from sklearn.metrics import roc_auc_score
from asd_methods import select_asd
from arcface import ArcFace
from util import (
    file_emb_new,
    saveModels,
    saveScoresAndDrawFig,
    count_params,
    draw_curve,
    draw_curves,
    stepDataloader,
    write2file,
)
from datasets.machineData1d23 import (
    MCMDataSet1d23,
    INVERSE_CLASS_MAP
)
from mobilenetv2 import MobileNetV2
import torchaudio
from tqdm import tqdm
import torchinfo
import logging
import glob
from collections import defaultdict


def train_and_test(
    net, trainds,
    emb_trainds,
    testds,
    ensemble_mode,
    asd_mode, args
):
    rmaxall = defaultdict(dict)
    result_curves_all = defaultdict(dict)
    loss_when_best_all = defaultdict(dict)
    learning_curves = []
    opt = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, net.parameters()),
        lr=args.learning_rate
    )
    lr_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=5)
    traindl = stepDataloader(trainds, args.batch_size, total_step=args.max_step)
    pbar = tqdm(traindl, total=args.max_step)
    obInterval = 200
    try:
        for curstep, medata in pbar:
            x = medata['observations'].cuda()
            label = medata['classify_labs'].cuda()
            loss = net(x, label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            learning_curves.append(loss.item())
            pbar.set_description(
                'step: {}/{} loss: {}'.format(
                    curstep, args.max_step, loss.item()
                )
            )
            if curstep % obInterval == 0:
                avg_loss = np.mean(learning_curves[-obInterval:])
                lr_sch.step(avg_loss)
                msg = "curstep: {}, current lr: {}".format(curstep, opt.param_groups[0]['lr'])
                print(msg)
                logging.info(msg)
                rnps, methods_names, scores_bufs = adall(
                    net, emb_trainds,
                    testds,
                    args.batch_size,
                    ensemble_mode,
                    asd_mode
                )
                for mt in rnps.keys():
                    rmax = rmaxall[mt]
                    result_curves = result_curves_all[mt]
                    loss_when_best = loss_when_best_all[mt]
                    rnp = rnps[mt]
                    methods_name = methods_names[mt]
                    scores_buf = scores_bufs[mt]
                    for i, tup in enumerate(methods_name):
                        if tup not in rmax.keys():
                            rmax[tup] = {'metric': np.zeros(3), 'buffer': None}
                            result_curves[tup] = []
                        temp = hmean(rnp[:, i])
                        result_curves[tup].append(temp)
                        if hmean(rmax[tup]['metric']) < temp:
                            rmax[tup]['metric'] = rnp[:, i]
                            loss_when_best[tup] = avg_loss
                            saveModels(net, tup, args)
                            rmax[tup]['buffer'] = scores_buf[i]
                net.train()
        draw_curve(learning_curves, os.path.join(
            args.save_path,
            f"{args.machine_type}_train_loss.jpg"
        ), savepickle=True)
        for mt, curves in result_curves_all.items():
            draw_curves(curves, os.path.join(
                args.save_path,
                f"{INVERSE_CLASS_MAP[mt]}_auc_pauc.jpg"
            ), savepickle=True)
    except Exception as err:
        print(err)
        logging.error(err)
    finally:
        if 'methods_names' in locals().keys():
            for mt in rmaxall.keys():
                args.mt = INVERSE_CLASS_MAP[mt]
                rmax = rmaxall[mt]
                loss_when_best = loss_when_best_all[mt]
                methods_name = methods_names[mt]
                msg = (
                    f"{args.netname} {count_params(net)} "
                    f"On {INVERSE_CLASS_MAP[mt]} best results:\n"
                )
                best_result = 0
                best_method = None
                for i, t in enumerate(methods_name):
                    msg += f'{t[0]}_{t[1]}: {rmax[t]["metric"][0]:.2f}/{rmax[t]["metric"][1]:.2f}/{rmax[t]["metric"][2]:.2f}; '
                    temp = hmean(rmax[t]["metric"], axis=None)
                    saveScoresAndDrawFig(rmax[t]['buffer'], args, draw=True)
                    if best_result < temp:
                        best_result = temp
                        best_method = t
                bestR = f'{rmax[best_method]["metric"][0]:.2f}/{rmax[best_method]["metric"][1]:.2f}/{rmax[best_method]["metric"][2]:.2f}'
                msg += '\n' + (
                    f'{INVERSE_CLASS_MAP[mt]} best results is: '
                    f'{best_method[0]}_{best_method[1]}; '
                    f'{bestR};'
                    f' loss_for_best={loss_when_best[best_method]:.3e};'
                    f' last_loss={avg_loss:.3e}'
                )
                print(msg)
                write2file(result_file, [INVERSE_CLASS_MAP[mt], bestR])
                logging.info(msg)


def adall(net, trainds, testds, bs, ensemble_mode: list, asd_mode: list):
    out_train = file_emb_new(net, trainds, bs, ensemble_mode)
    embs_list_train = out_train['observations']
    mts_train = out_train['machine_types']
    secs_train = out_train['machine_sections']
    out_test = file_emb_new(net, testds, bs, ensemble_mode)
    embs_list_test = out_test['observations']
    fids_test = out_test['file_ids']
    mts_test = out_test['machine_types']
    secs_test = out_test['machine_sections']
    doms_test = out_test['domains']
    y_test = out_test['targets']
    unimts = np.unique(mts_test)
    rnps = dict()
    methods_names = dict()
    scores_buffers = dict()
    for mt in unimts:
        unisecs = np.unique(secs_test[mts_test == mt])
        source_aucs = []
        target_aucs = []
        paucs = []
        methods_name = []
        scores_buffer = []
        error_methods = set()
        for asd in asd_mode:
            for embs_test, embs_train, emode in zip(
                embs_list_test, embs_list_train, ensemble_mode
            ):
                auc_source_per_method = []
                auc_target_per_method = []
                pauc_per_method = []
                sb_per_method = []
                for sec in unisecs:
                    if (asd, emode) in error_methods:
                        continue
                    index_test = np.logical_and(secs_test == sec, mts_test == mt)
                    index_train = np.logical_and(secs_train == sec, mts_train == mt)
                    try:
                        embs_test_unit = embs_test[index_test]
                        embs_train_unit = embs_train[index_train]
                        y_test_unit = y_test[index_test]
                        fids_test_unit = fids_test[index_test]
                        doms_test_unit = doms_test[index_test]
                        index_test_source = np.logical_or(y_test_unit == 1, doms_test_unit == 'source')
                        index_test_target = np.logical_or(y_test_unit == 1, doms_test_unit == 'target')
                        y_test_scores_unit = select_asd(
                            asd,
                            embs_test_unit,
                            embs_train_unit,
                        )
                        sb_per_method.append((
                            y_test_scores_unit, fids_test_unit,
                            y_test_unit, embs_test_unit,
                            sec, asd, emode
                        ))
                        auc_source_per_method.append(
                            roc_auc_score(
                                y_test_unit[index_test_source],
                                y_test_scores_unit[index_test_source]
                            )
                        )
                        auc_target_per_method.append(
                            roc_auc_score(
                                y_test_unit[index_test_target],
                                y_test_scores_unit[index_test_target]
                            )
                        )
                        pauc_per_method.append(
                            roc_auc_score(
                                y_test_unit,
                                y_test_scores_unit,
                                max_fpr=0.1
                            )
                        )
                    except Exception as err:
                        print(f"error: {err} ocur in {asd}_{emode}")
                        error_methods.add((asd, emode))
                if (asd, emode) not in error_methods:
                    methods_name.append((asd, emode))
                    source_aucs.append(auc_source_per_method)
                    target_aucs.append(auc_target_per_method)
                    paucs.append(pauc_per_method)
                    scores_buffer.append(sb_per_method)
        source_aucs, target_aucs, paucs = map(np.array, [source_aucs, target_aucs, paucs])
        source_aucs = source_aucs * 100
        target_aucs = target_aucs * 100
        paucs = paucs * 100
        source_auc = hmean(source_aucs, axis=1)
        target_auc = hmean(target_aucs, axis=1)
        pauc = hmean(paucs, axis=1)
        msg = f"On {INVERSE_CLASS_MAP[mt]} Test Data:\n"
        for i, t in enumerate(methods_name):
            msg += f'{t[0]}_{t[1]}: {source_auc[i]:.2f}/{target_auc[i]:.2f}/{pauc[i]:.2f}; '
        print(msg)
        rnps[mt] = np.stack([source_auc, target_auc, pauc])
        methods_names[mt] = methods_name
        scores_buffers[mt] = scores_buffer
    return rnps, methods_names, scores_buffers


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mt', '--machine_type', type=int, default=-1)
    parser.add_argument('-bs', '--batch_size', type=int, default=150)
    parser.add_argument('-ep', '--exp_path', type=str, default=exp)
    parser.add_argument(
        '-lr', '--learning_rate',
        type=float, default=1e-3,
        help="learning rate"
    )
    parser.add_argument('--max_step', type=int, default=10000)
    parser.add_argument('--result_file', type=str)
    parser.add_argument('--task', type=str, default='machine') # or 'condition'
    args = parser.parse_args()
    expkw = args.exp_path #'classify'
    machine_type = args.machine_type
    result_file = args.result_file
    if machine_type == -1:
        args.machine_type = 'allmachines'
    else:
        args.machine_type = INVERSE_CLASS_MAP[machine_type]
    logfile = f'logs/{expkw}_log_{args.machine_type}_*.log'
    logfile = logfile.replace('*', str(len(glob.glob(logfile))+1))
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    logging.basicConfig(
        filename=logfile,
        format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        level=logging.INFO,
    )
    # 准备数据部分
    sr = 16000
    n_fft = 1024
    hop_length = n_fft // 2
    n_mels = 128
    n_frames = 186
    input_samples = n_fft + (n_frames - 1) * hop_length
    hop_size = None
    mcm = MCMDataSet1d23(
        machine_types=machine_type,
        input_samples=input_samples,
        hop_size=hop_size,
        task=args.task
    )
    trainds = mcm.training_data_set()
    emb_trainds = mcm.embedding_data_set()
    testds = mcm.validation_data_set()

    # 准备网络部分，网络中包含损失函数，训练时网络返回loss，测试时网络返回embedding
    n_classes = mcm.n_classes
    emb_size = 128
    featmodule = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, center=False)
    lossfn = ArcFace(emb_size, n_classes)
    net = MobileNetV2(
        emb_size=emb_size,
        featmodule=featmodule,
        lossfn=lossfn,
    )
    net.cuda()
    logging.info("featmodule: "+featmodule.__class__.__name__)
    logging.info("loss function: "+lossfn.__class__.__name__)
    netinfo = torchinfo.summary(net, input_size=(input_samples,), batch_dim=0, depth=2)
    logging.info('network summary:\n'+str(netinfo))
    args.netname = net.__class__.__name__
    args.save_path = f"results23{expkw}_{args.task}"
    os.makedirs(args.save_path, exist_ok=True)
    ensemble_mode = ['mean', 'std']
    asd_mode = ['lof', 'maha', 'knn', 'cos']
    train_and_test(
        net,
        trainds,
        emb_trainds,
        testds,
        ensemble_mode,
        asd_mode, args
    )

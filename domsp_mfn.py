import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import argparse
import numpy as np
import pandas as pd
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
    rnps, methods_names, scores_bufs = adall(
        net, emb_trainds,
        testds,
        args.batch_size,
        ensemble_mode,
        asd_mode,
        args.domsp,
    )
    best_result = {'result': [], 'method': []}
    df = pd.DataFrame({f'{a}-{e}': [] for a in asd_mode for e in ensemble_mode})
    for mt in rnps.keys():
        rnp = rnps[mt]
        mt_method_hm = hmean(rnp, axis=0)  # 3*8 => 8
        mt_bm = np.argmax(mt_method_hm)  # best_method
        mt_br = mt_method_hm[mt_bm]  # best_result
        best_result['result'].append(rnp[:, mt_bm].tolist() + [mt_br])
        best_result['method'].append(methods_names[mt][mt_bm])
        for i in range(3):
            df.loc[len(df)] = np.round(rnp[i, :], 2)
        df.loc[len(df)] = np.round(mt_method_hm, 2)
    df.loc[len(df)] = np.round(hmean(np.array(df), axis=0), 2)
    full_result = np.array(best_result['result'])
    all_hmean = hmean(full_result[:, -1])
    best_result['all_hmean'] = all_hmean
    metric_hmean = np.round(hmean(full_result, axis=0), 2)
    save_result = np.round(full_result, 2)
    save_result = np.vstack([save_result, metric_hmean[np.newaxis, :]])
    bdf = pd.DataFrame({n: save_result[:, i] for i, n in enumerate(['s_auc', 't_auc', 'pauc', 'hmean'])})
    bdf['detector'] = [f'{a[0]}-{a[1]}' for a in best_result['method']] + ['full']
    df.to_csv(f'./direct_infer_result/mfn_{args.domsp}_all_method.csv')
    bdf.to_csv(f'./direct_infer_result/mfn_{args.domsp}_best_detector.csv')

    best_backend = df.loc[len(df) - 1].idxmax()  # 最好的后端
    result_path = os.path.join(args.result_path, args.domsp)
    os.makedirs(result_path, exist_ok=True)
    infer_scores(best_backend, args.domsp, result_path)


def adall(net, trainds, testds, bs, ensemble_mode, asd_mode, domsp):
    out_tr_te_fn = './direct_infer_ckpt/mfn_out.npz'
    if not os.path.exists(out_tr_te_fn):
        out_train = file_emb_new(net, trainds, bs, ensemble_mode)
        out_test = file_emb_new(net, testds, bs, ensemble_mode)
        np.savez(out_tr_te_fn, out_train=out_train, out_test=out_test)
    else:
        out_tr_te = np.load(out_tr_te_fn, allow_pickle=True)
        out_train = out_tr_te['out_train'].item()
        out_test = out_tr_te['out_test'].item()
    embs_list_train = out_train['observations']
    mts_train = out_train['machine_types']
    secs_train = out_train['machine_sections']
    doms_train = out_train['domains']
    embs_list_test = out_test['observations']
    fids_test = out_test['file_ids']
    mts_test = out_test['machine_types']
    secs_test = out_test['machine_sections']
    doms_test = out_test['domains']
    dom_pred = out_test['dom_pred']
    y_test = out_test['targets']
    unimts = np.sort(np.unique(mts_test)) % 7
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
                    mt_dom_pred = dom_pred[index_test]
                    try:
                        embs_test_unit = embs_test[index_test]
                        y_test_unit = y_test[index_test]
                        fids_test_unit = fids_test[index_test]
                        doms_test_unit = doms_test[index_test]
                        if domsp == 'none':
                            index_train = np.logical_and(secs_train == sec, mts_train == mt)
                            embs_train_unit = embs_train[index_train]  # 1000*128
                        elif domsp in ['hard', 'soft', 'min']:
                            index_train = {dom: np.logical_and(secs_train == sec, np.logical_and(mts_train == mt, doms_train == dom))
                                           for dom in ['source', 'target']}
                            embs_train_unit = {dom: embs_train[indices] for dom, indices in index_train.items()}
                        index_test_source = np.logical_or(y_test_unit == 1, doms_test_unit == 'source')
                        index_test_target = np.logical_or(y_test_unit == 1, doms_test_unit == 'target')
                        if domsp == 'none':
                            y_test_scores_unit = select_asd(
                                asd,
                                embs_test_unit,
                                embs_train_unit,)
                        elif domsp in ['hard', 'soft', 'min']:
                            y_test_scores_unit_source = select_asd(  # 所有该类型clip的score
                                asd,
                                embs_test_unit,
                                embs_train_unit['source'],)
                            y_test_scores_unit_source = (y_test_scores_unit_source - np.mean(y_test_scores_unit_source)) / np.std(y_test_scores_unit_source)
                            y_test_scores_unit_target = select_asd(  # 所有该类型clip的score
                                asd,
                                embs_test_unit,
                                embs_train_unit['target'],)
                            y_test_scores_unit_target = (y_test_scores_unit_target - np.mean(y_test_scores_unit_target)) / np.std(y_test_scores_unit_target)
                            y_test_scores_unit = np.vstack([y_test_scores_unit_source,
                                                            y_test_scores_unit_target]).T  # 200 * 2
                            if domsp == 'hard':
                                thres_tar = 0.5
                                dom_coeff = np.vstack([np.ones(y_test_scores_unit.shape[0]),
                                                       np.zeros(y_test_scores_unit.shape[0])]).T  # 200 *2
                                pred_tar = np.where(mt_dom_pred[:, 1] > thres_tar)[0]
                                dom_coeff[pred_tar] = np.array([0, 1])
                                y_test_scores_unit = (y_test_scores_unit * dom_coeff).sum(axis=-1)
                            elif domsp == 'soft':
                                dom_coeff = mt_dom_pred
                                y_test_scores_unit = (y_test_scores_unit * dom_coeff).sum(axis=-1)
                            elif domsp == 'min':
                                y_test_scores_unit = y_test_scores_unit.min(axis=-1)

                        sb_per_method.append((
                            y_test_scores_unit, fids_test_unit,
                            y_test_unit, embs_test_unit,
                            sec, asd, emode))
                        auc_source_per_method.append(
                            roc_auc_score(
                                y_test_unit[index_test_source],
                                y_test_scores_unit[index_test_source]))
                        auc_target_per_method.append(
                            roc_auc_score(
                                y_test_unit[index_test_target],
                                y_test_scores_unit[index_test_target]))
                        pauc_per_method.append(
                            roc_auc_score(
                                y_test_unit,
                                y_test_scores_unit,
                                max_fpr=0.1))
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


def infer_scores(backend, domsp, result_path):
    best_asd, best_emode = backend.split('-')
    out_tr_te_fn = './direct_infer_ckpt/mfn_out.npz'
    out_tr_te = np.load(out_tr_te_fn, allow_pickle=True)
    out_train = out_tr_te['out_train'].item()
    out_test = out_tr_te['out_test'].item()
    embs_list_train = out_train['observations']
    mts_train = out_train['machine_types']
    secs_train = out_train['machine_sections']
    doms_train = out_train['domains']
    embs_list_test = out_test['observations']
    fids_test = out_test['file_ids']
    mts_test = out_test['machine_types']
    secs_test = out_test['machine_sections']
    dom_pred = out_test['dom_pred']
    unimts = np.sort(np.unique(mts_test))
    for mt in unimts:
        unisecs = np.unique(secs_test[mts_test == mt])
        for embs_test, embs_train, emode in zip(
            embs_list_test, embs_list_train, ensemble_mode
        ):
            if not (emode == best_emode):
                continue
            for sec in unisecs:
                index_test = np.logical_and(secs_test == sec, mts_test == mt)
                index_train = np.logical_and(secs_train == sec, mts_train == mt)
                mt_dom_pred = dom_pred[index_test]
                embs_test_unit = embs_test[index_test]
                fids_test_unit = fids_test[index_test]
                if domsp == 'none':
                    index_train = np.logical_and(secs_train == sec, mts_train == mt)
                    embs_train_unit = embs_train[index_train]  # 1000*128
                elif domsp in ['hard', 'soft', 'min']:
                    index_train = {dom: np.logical_and(secs_train == sec, np.logical_and(mts_train == mt, doms_train == dom))
                                   for dom in ['source', 'target']}
                    embs_train_unit = {dom: embs_train[indices] for dom, indices in index_train.items()}
                if domsp == 'none':
                    y_test_scores_unit = select_asd(
                        best_asd,
                        embs_test_unit,
                        embs_train_unit,)
                elif domsp in ['hard', 'soft', 'min']:
                    y_test_scores_unit_source = select_asd(  # 所有该类型clip的score
                        best_asd,
                        embs_test_unit,
                        embs_train_unit['source'],)
                    y_test_scores_unit_source = (y_test_scores_unit_source - np.mean(y_test_scores_unit_source)) / np.std(y_test_scores_unit_source)
                    y_test_scores_unit_target = select_asd(  # 所有该类型clip的score
                        best_asd,
                        embs_test_unit,
                        embs_train_unit['target'],)
                    y_test_scores_unit_target = (y_test_scores_unit_target - np.mean(y_test_scores_unit_target)) / np.std(y_test_scores_unit_target)
                    y_test_scores_unit = np.vstack([y_test_scores_unit_source,
                                                    y_test_scores_unit_target]).T  # 200 * 2
                    if domsp == 'hard':
                        thres_tar = 0.5
                        dom_coeff = np.vstack([np.ones(y_test_scores_unit.shape[0]),
                                               np.zeros(y_test_scores_unit.shape[0])]).T  # 200 *2
                        pred_tar = np.where(mt_dom_pred[:, 1] > thres_tar)[0]
                        dom_coeff[pred_tar] = np.array([0, 1])
                        y_test_scores_unit = (y_test_scores_unit * dom_coeff).sum(axis=-1)
                    elif domsp == 'soft':
                        dom_coeff = mt_dom_pred
                        y_test_scores_unit = (y_test_scores_unit * dom_coeff).sum(axis=-1)
                    elif domsp == 'min':
                        y_test_scores_unit = y_test_scores_unit.min(axis=-1)
                mt_df = pd.DataFrame({'file_ids': [os.path.basename(f) for f in fids_test_unit.tolist()],
                                      'scores': y_test_scores_unit.tolist()})
                mt_df.to_csv(os.path.join(result_path, F'anomaly_score_{INVERSE_CLASS_MAP[mt]}_section_{sec}.csv'),
                             header=False, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mt', '--machine_type', type=int, default=-1)  # -1为dev+eval，-2为dev，-3为eval
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-dur', '--input_duration', type=float, default=2)
    parser.add_argument('-ac', '--accumulate_grad', type=int, default=8)
    parser.add_argument('-ep', '--exp_path', type=str, default="exp")
    parser.add_argument('-sp', '--speed_perturb', type=int, default=0)
    parser.add_argument('-interval', '--obInterval', type=int, default=500)
    parser.add_argument(
        '-lr', '--learning_rate',
        type=float, default=1e-3,
        help="learning rate"
    )
    parser.add_argument('--max_step', type=int, default=10000)
    parser.add_argument('--result_file', type=str)
    parser.add_argument('--task', type=str, default='condition')  # or 'condition'
    parser.add_argument('--domsp', choices=['none', 'hard', 'soft', 'min'], default='none')
    parser.add_argument('--result_path', type=str, default='csv_path/mfn')
    args = parser.parse_args()
    expkw = args.exp_path  # 'classify'
    machine_type = args.machine_type
    result_file = args.result_file
    if machine_type == -1:
        args.machine_type = 'allmachines'
    elif machine_type == -2:
        args.machine_type = 'devmachines'
    elif machine_type == -3:
        args.machine_type = 'evalmachines'
    else:
        args.machine_type = INVERSE_CLASS_MAP[machine_type]
    logfile = f'logs/{expkw}_log_{args.machine_type}_*.log'
    logfile = logfile.replace('*', str(len(glob.glob(logfile)) + 1))
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
    input_samples = int(sr * args.input_duration)
    hop_size = None
    mcm = MCMDataSet1d23(
        data_root="/home/public/dcase/dcase23/updated/",
        machine_types=machine_type,
        input_samples=input_samples,
        hop_size=hop_size,
        task=args.task,
        sp=bool(args.speed_perturb)
    )
    trainds = mcm.training_data_set()
    emb_trainds = mcm.embedding_data_set()
    testds = mcm.validation_data_set()
    # 准备网络部分，网络中包含损失函数，训练时网络返回loss，测试时网络返回embedding
    # n_classes = mcm.n_classes
    n_classes = 167
    emb_size = 128
    featmodule = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
                                                      n_mels=n_mels, center=False)
    lossfn = ArcFace(emb_size, n_classes)
    net = MobileNetV2(
        emb_size=emb_size,
        featmodule=featmodule,
        lossfn=lossfn,
    )
    m = 'results23exp_mobilefacenet_2s_eval_0_condition_step_1600_method_lof_mean_hmean_58.043'
    net_dict = torch.load(f'./best_ckpts/{m}',
                          map_location='cpu')
    net.load_state_dict(net_dict)
    net.cuda()
    logging.info("featmodule: " + featmodule.__class__.__name__)
    logging.info("loss function: " + lossfn.__class__.__name__)
    netinfo = torchinfo.summary(net, input_size=(input_samples,), batch_dim=0, depth=2)
    logging.info('network summary:\n' + str(netinfo))
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

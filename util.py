import os
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from sklearn.manifold import TSNE
from collections import defaultdict
import pickle


plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Times New Roman']
plt.rcParams['font.size'] = 15


@torch.no_grad()
def calc_embs(net, ds, bs):
    testdl = DataLoader(ds, bs, shuffle=False)
    net.eval()
    embs = []
    secs = []
    labs = []
    fids = []
    mts = []
    for medata in tqdm(testdl):
        x = medata['observations'].cuda()
        embs.append(net(x)['embedding'].cpu().numpy())
        secs.append(medata['machine_sections'].numpy())
        labs.append(medata['targets'].numpy())
        fids.extend(medata['file_ids'])
        mts.append(medata['machine_types'].numpy())
    embeddings = np.concatenate(embs, axis=0)
    sections = np.concatenate(secs, axis=0)
    labels = np.concatenate(labs, axis=0)
    fileids = np.array(fids)
    mtypes = np.concatenate(mts, axis=0)
    return embeddings, sections, labels, fileids, mtypes


@torch.no_grad()
def calc_embs_new(net, ds, bs, specialKey):  # specialKey: 'observations'
    out = {k: list() for k in ds[0].keys()}
    testdl = DataLoader(ds, bs, shuffle=False)
    net.eval()
    for medata in tqdm(testdl):
        x = medata[specialKey].cuda()
        out[specialKey].append(net(x)['embedding'].cpu().numpy())
        for k, v in out.items():
            if k == specialKey:
                continue
            if type(medata[k]) is torch.Tensor:
                v.append(medata[k].numpy())
            else:
                v.append(medata[k])
    for k, v in out.items():
        out[k] = np.concatenate(v, axis=0)
    return out


def ensemble_embs(embs, mode):
    if mode == 'mean':
        r = np.mean(embs, axis=0)
    elif mode == 'std':
        r = np.std(embs, axis=0)
    elif mode == 'mean_std':
        r = np.concatenate([np.mean(embs, axis=0), np.std(embs, axis=0)])
    elif mode == 'max':
        r = np.max(embs, axis=0)
    elif mode == 'min':
        r = np.min(embs, axis=0)
    else:
        # 暂时没有
        raise AttributeError
    return r


def to_per_file(embs, secs, labs, fids, mts, ensemble_mode):
    if isinstance(ensemble_mode, str):
        ensemble_mode = [ensemble_mode]
    # 感觉return_index返回的索引是每个uniq值第一次出现的位置
    unifiles, firstinds, counts = np.unique(fids, return_index=True, return_counts=True)
    sec_per_file = []
    lab_per_file = []
    embs_list = []
    mt_per_file = []
    for i, mode in enumerate(ensemble_mode):
        emb_per_file = []
        for j, ind1st in enumerate(firstinds):
            ind = slice(ind1st, ind1st + counts[j])
            emb_per_file.append(ensemble_embs(embs[ind], mode))
            if i == 0:
                sec_per_file.append(secs[ind1st])
                lab_per_file.append(labs[ind1st])
                mt_per_file.append(mts[ind1st])
        emb_per_file = np.array(emb_per_file)
        embs_list.append(emb_per_file)
    secs, labs, mts = map(np.array, [sec_per_file, lab_per_file, mt_per_file])
    return embs_list, secs, labs, unifiles, mts


def to_per_file_new(dataDict, mainKey, ensemble_mode, specialKey):
    out = {k: list() for k in dataDict.keys()}
    if isinstance(ensemble_mode, str):
        ensemble_mode = [ensemble_mode]
    # 感觉return_index返回的索引是每个uniq值第一次出现的位置
    _, firstinds, counts = np.unique(dataDict[mainKey], return_index=True, return_counts=True)
    embsDict = defaultdict(list)
    for j, ind1st in enumerate(firstinds):
        ind = slice(ind1st, ind1st + counts[j])
        for k, v in out.items():
            if k == specialKey:
                continue
            v.append(dataDict[k][ind1st])
        for mode in ensemble_mode:
            embsDict[mode].append(ensemble_embs(dataDict[specialKey][ind], mode))
    out[specialKey] = [np.array(v) for v in embsDict.values()]
    for k, v in out.items():
        if k == specialKey:
            continue
        out[k] = np.array(v)
    return out


def file_emb(net, ds, bs, ensemble_mode: list = ['mean', 'std']):
    embs, secs, labs, fids, mts = calc_embs(net, ds, bs)
    embs_list, secs, labs, fids, mts = to_per_file(
        embs, secs, labs, fids, mts, ensemble_mode
    )
    return embs_list, secs, labs, fids, mts


def file_emb_new(
    net, ds, bs,
    ensemble_mode: list = ['mean', 'std'],
    mainKey: str = 'file_ids',
    specialKey: str = 'observations',
):
    hdict = calc_embs_new(net, ds, bs, specialKey)
    outdic = to_per_file_new(hdict, mainKey, ensemble_mode, specialKey)
    return outdic


def saveModels(net, method: tuple, args, mt, br=None):
    filepath = os.path.join(args.save_path, "saved_models")
    os.makedirs(filepath, exist_ok=True)
    temp = os.listdir(filepath)
    filename = f'{args.machine_type}_{method[0]}_{method[1]}_{mt}'
    saveflag = True
    if br is not None:
        for it in temp:
            itf = os.path.split(it)[1]
            if itf.startswith(filename):
                oldr = itf.split('_')[3]
                try:
                    oldr = float(oldr)
                except Exception:
                    oldr = None
                if type(oldr) is float:
                    if oldr > br:
                        saveflag = False
                    else:
                        os.remove(os.path.join(filepath, it))
    if saveflag:
        if br is not None:
            filename += f'_{br:.2f}_statedic.pkl'
        else:
            filename += '_statedic.pkl'
        filepath = os.path.join(filepath, filename)
        torch.save(net.state_dict(), filepath)


def saveScoresAndDrawFig(
    scores_buffer: List[tuple], args, draw=False
):
    '''
    scores_buffer: list[(scores_per_sec, fids_per_sec,
    target_per_sec, embs_per_sec,
    sec, asd_mode, ensemble_mode),(...),...,(...)]
    method: (asd_mode, ensemble_mode)
    '''
    dependic = defaultdict(list)
    embsdic = defaultdict(list)
    for scores_tup in scores_buffer:
        labels = scores_tup[2]
        embs = scores_tup[3]
        sec = scores_tup[4]
        asd = scores_tup[5]
        emode = scores_tup[6]
        dependic[f'{asd}_{emode}'].extend([f'ID{sec}_{"异常" if label==1 else "正常"}' for label in labels])
        embsdic[f'{asd}_{emode}'].append(embs)
        dependon = np.array(['异常' if label else '正常' for label in labels])
        write_scores(
            args,
            scores=scores_tup[0],
            fileids=scores_tup[1],
            dependon=dependon,
            sec=sec,
            asd=asd,
            emode=emode,
            draw=draw
        )
        if draw:
            filename = f'tsne_{args.mt}_section_{sec:02d}_{asd}_{emode}.jpg'
            filepath = os.path.join(args.save_path, "tsnefigs", filename)
            drawTSNEfig(embs, dependon, filepath)
    if draw:
        for key in dependic.keys():
            filepath = os.path.join(args.save_path, "tsnefigs", f'tsne_{args.mt}_{key}.jpg')
            allembs = np.concatenate(embsdic[key], axis=0)
            alldepend = np.array(dependic[key])
            drawTSNEfig(allembs, alldepend, filepath)


def drawTSNEfig(embs, dependon, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    points = TSNE(
        n_components=2,
        random_state=66,
        learning_rate="auto",
        init="pca",
        metric="cosine"
    ).fit_transform(embs)
    np.savez(str.rsplit(filepath, '.', 1)[0]+'.npz', x=points, y=dependon)
    scatterDependonItem(
        points, dependon,
        figname=filepath
    )


def scatterDependonItem(embs, item, figname):
    plt.figure(figsize=(8, 6), dpi=150)
    uni = np.unique(item)
    for i in uni:
        ind = item == i
        x = embs[ind][:, 0]
        y = embs[ind][:, 1]
        plt.scatter(x, y, alpha=0.5, label=str(i))
    plt.legend(loc='best')
    plt.savefig(figname)
    plt.close()


def write_scores(
    args,
    fileids, scores, dependon,
    sec, asd, emode, draw=False
):
    assert len(fileids) == len(scores)
    filename = 'anomaly_score_{}_section_{:02d}_{}_{}.csv'.format(
        args.machine_type,
        sec,
        asd,
        emode,
    )
    filepath = os.path.join(args.save_path, "scores", filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pd.DataFrame(
        list(zip(fileids, scores))
    ).to_csv(filepath, index=False, header=False)
    if draw:
        figname = filepath.rsplit('.', 1)[0] + '.jpg'
        drawscores(scores, dependon, figname)


def drawscores(scores, dependon, figname):
    '''
    scores (np.array) : 异常分数
    figname (str) : 保存图片的路径名字
    '''
    plt.figure(figsize=(8, 6), dpi=150)
    unidep = np.unique(dependon)
    bins = np.linspace(scores.min(), scores.max(), 50)
    for item in unidep:
        item_scores = scores[dependon == item]
        plt.hist(item_scores, bins, alpha=0.3, label=item)
    plt.legend(loc='best')
    plt.savefig(figname)
    plt.close()


def draw_curves(curves, savepath, savepickle=True):
    if savepickle:
        picklepath = str.rsplit(savepath, '.', 1)[0] + '.pickle'
        writeToPickle(curves, picklepath)
    plt.figure(figsize=(8, 6), dpi=150)
    for k, v in curves.items():
        plt.plot(v, label=str(k))
    plt.legend(bbox_to_anchor=(1, 0), loc=3, borderaxespad=0)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def draw_curve(curve, savepath, savepickle=True):
    if savepickle:
        picklepath = str.rsplit(savepath, '.', 1)[0] + '.pickle'
        writeToPickle(curve, picklepath)
    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot(curve)
    plt.savefig(savepath)
    plt.close()


def writeToPickle(obj, picklepath):
    with open(picklepath, "wb") as wfile:
        pickle.dump(obj, wfile)


def loadFromPickle(picklepath):
    with open(picklepath, "rb") as wfile:
        obj = pickle.load(wfile)
        return obj


class net_wrapper(torch.nn.Module):
    def __init__(self, spec_module, core_net) -> None:
        super().__init__()
        self.spec_module = spec_module
        self.core_net = core_net

    def forward(self, x, label=None):
        spec = self.spec_module(x)
        out = self.core_net(spec, label)
        return out


def count_params(model):
    num_param = 0
    for param in model.parameters():
        num_param += param.numel()
    msg = 'model size: {:.2f} M'.format(num_param/1000/1000)
    return msg


def fmtdic(d):
    temp = [f'{k}:{v}' for k, v in d.items()]
    temp = sorted(temp)
    out = '-'.join(temp)
    return out


def stepDataloader(ds, batch_size, total_step=10000):
    curstep = 0
    while True:
        dl = DataLoader(ds, batch_size, shuffle=True)
        for item in dl:
            curstep += 1
            if curstep <= total_step:
                yield curstep, item
            else:
                return


def write2file(filepath, line):
    with open(filepath, 'a') as wf:
        for i, item in enumerate(line):
            wf.write(str(item))
            if i + 1 < len(line):
                wf.write(',')
            else:
                wf.write('\n')


def statResults(filepath, outfile):
    x = pd.read_csv(filepath, header=None).to_numpy()
    out = defaultdict(list)
    for line in x:
        mt = line[0]
        r = [float(i) for i in line[1].split('/')]
        out[mt].append(r)
    for k, v in out.items():
        v = np.array(v)
        mu = np.mean(v, axis=0)
        std = np.std(v, axis=0)
        outline = [k] + np.array([mu, std]).T.reshape(-1).tolist()
        write2file(outfile, outline)

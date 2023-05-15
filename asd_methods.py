'''
通过embedding计算异常分数的各种方法
每个方法返回的是一个异常分数的np.array
'''
import torch
import torch.nn.functional as F
import numpy as np
# from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import DistanceMetric
from pyod.models.lof import LOF
# from pyod.models.knn import KNN
from pyod.models.mcd import MCD
from pyod.models.cof import COF
from pyod.models.pca import PCA
from pyod.models.ocsvm import OCSVM
from pyod.models.lmdd import LMDD
from pyod.models.cblof import CBLOF

# def lof_score(embs_test, embs_train):
#     clf = LocalOutlierFactor(
#         n_neighbors=4,
#         novelty=True,
#         # metric='cosine'
#     )
#     clf.fit(embs_train)
#     scores = -clf.decision_function(embs_test)
#     return scores


def knn_score(embs_test, embs_train):
    clf = NearestNeighbors(
        n_neighbors=1,
        metric='cosine'
    )
    clf.fit(embs_train)
    scores = clf.kneighbors(embs_test, 1)[0].reshape(-1)
    return scores


def lof_score(embs_test, embs_train):
    clf = LOF(
        n_neighbors=4,
        contamination=1e-3,
        metric='cosine'
    )
    clf.fit(embs_train)
    scores = clf.decision_function(embs_test)
    return scores


# def knn_score(embs_test, embs_train):
#     clf = KNN(n_neighbors=1, metric='cosine', contamination=0.01)
#     clf.fit(embs_train)
#     scores = clf.decision_function(embs_test)
#     return scores


def mcd_score(embs_test, embs_train):
    clf = MCD(contamination=0.01)
    clf.fit(embs_train)
    scores = clf.decision_function(embs_test)
    return scores


def cof_score(embs_test, embs_train):
    clf = COF(contamination=0.01, n_neighbors=4)
    clf.fit(embs_train)
    scores = clf.decision_function(embs_test)
    return scores


def pca_score(embs_test, embs_train):
    clf = PCA(contamination=0.01)
    clf.fit(embs_train)
    scores = clf.decision_function(embs_test)
    return scores


def ocsvm_score(embs_test, embs_train):
    clf = OCSVM(contamination=0.01)
    clf.fit(embs_train)
    scores = clf.decision_function(embs_test)
    return scores


def lmdd_score(embs_test, embs_train):
    clf = LMDD(contamination=0.01)
    clf.fit(embs_train)
    scores = clf.decision_function(embs_test)
    return scores


def cblof_score(embs_test, embs_train):
    clf = CBLOF(contamination=0.01, n_clusters=3)
    clf.fit(embs_train)
    scores = clf.decision_function(embs_test)
    return scores


def maha_score(embs_test, embs_train):
    if np.isnan(embs_test).sum() + np.isnan(embs_train).sum() > 0:
        raise ValueError("there is nan in embs")
    try:
        if embs_train.shape[0] <= embs_train.shape[1]:
            print(embs_train.shape[0], embs_train.shape[1])
            print('观测样本数少于特征维度数，协方差矩阵不可逆！')
        mean_emb_per_sec = np.mean(
            embs_train,
            axis=0
        )
        cov_per_sec = np.cov(
            embs_train,
            rowvar=False
        )
        if np.isnan(cov_per_sec).sum() > 0:
            raise ValueError("there is nan in the cov of train_embs")
        cov_per_sec += 1e-6 * np.eye(cov_per_sec.shape[0])
        dist = DistanceMetric.get_metric(
            'mahalanobis', V=cov_per_sec
        )
        scos_per_sec = []
        for emb in embs_test:
            td = dist.pairwise([emb, mean_emb_per_sec])[0][1]
            if not np.isnan(td):
                scos_per_sec.append(td)
            else:
                print(td)
                print(emb, mean_emb_per_sec)
                print('there is a nan!')
                raise ValueError
        scos_per_sec = np.array(scos_per_sec)
        return scos_per_sec
    except Exception as err:
        print("ocur error when calculate mahanobis distance")
        raise err


def cos_score(embs_test, embs_train):
    mean_emb_per_sec = torch.from_numpy(
        np.mean(embs_train, axis=0)
    ).repeat(embs_test.shape[0], 1)
    embs_test = torch.from_numpy(embs_test)
    return 1 - torch.cosine_similarity(
        embs_test, mean_emb_per_sec
    ).numpy()


@torch.no_grad()
def prob_score(embs_test, sfw, sec: int):
    '''
    embs_test is np.array
    '''
    embs_test = torch.from_numpy(embs_test).to(sfw.device)
    # logits = F.linear(F.normalize(embs_test), F.normalize(sfw))
    logits = F.linear(F.normalize(embs_test), sfw)
    # logits = F.linear(embs_test, sfw)
    probs = F.softmax(logits, dim=1)
    prob = probs[:, sec]
    scores = torch.log((1 - prob) / (prob + 1e-9) + 1e-15).cpu().numpy()
    return scores


def select_asd(*args):
    '''
    args[0]: asd
    args[1]: embs_test
    args[2]: embs_train
    '''
    if args[0] == 'knn':
        scores = knn_score(args[1], args[2])
    elif args[0] == 'lof':
        scores = lof_score(args[1], args[2])
    elif args[0] == 'cof':
        scores = cof_score(args[1], args[2])
    elif args[0] == 'mcd':
        scores = mcd_score(args[1], args[2])
    elif args[0] == 'pca':
        scores = pca_score(args[1], args[2])
    elif args[0] == 'ocsvm':
        scores = ocsvm_score(args[1], args[2])
    elif args[0] == 'lmdd':
        scores = lmdd_score(args[1], args[2])
    elif args[0] == 'cblof':
        scores = cblof_score(args[1], args[2])
    elif args[0] == 'cos':
        scores = cos_score(args[1], args[2])
    elif args[0] == 'maha':
        scores = maha_score(args[1], args[2])
    else:
        raise AttributeError
    return scores

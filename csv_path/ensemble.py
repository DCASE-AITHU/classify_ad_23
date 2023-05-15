import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import hmean
from matplotlib import cm
from sklearn.metrics import roc_auc_score
from mpl_toolkits.mplot3d import Axes3D


DEV_TYPES23 = [
    'bearing',
    'fan',
    'gearbox',
    'slider',
    'ToyCar',
    'ToyTrain',
    'valve',
]
EVAL_TYPES23 = [
    'bandsaw',
    'grinder',
    'shaker',
    'ToyDrone',
    'ToyNscale',
    'ToyTank',
    'Vacuum'
]
ALL_TYPES23 = DEV_TYPES23 + EVAL_TYPES23
CLASS_MAP23 = {
    'bearing': 0,
    'fan': 1,
    'gearbox': 2,
    'slider': 3,
    'ToyCar': 4,
    'ToyTrain': 5,
    'valve': 6,
    'bandsaw': 7,
    'grinder': 8,
    'shaker': 9,
    'ToyDrone': 10,
    'ToyNscale': 11,
    'ToyTank': 12,
    'Vacuum': 13
}
INVERSE_CLASS_MAP23 = {v: k for k, v in CLASS_MAP23.items()}


class GridSearch:
    def __init__(self, score_dirs, delta):
        self.model = list(score_dirs.keys())
        self.model_num = len(self.model)
        self.delta = delta
        all_file_dfs = {mt: [] for mt in ALL_TYPES23}
        for model, sdir in score_dirs.items():
            files = glob.glob(os.path.join(sdir, '*.csv'))
            for mt_file in files:
                mt = os.path.basename(mt_file).split('_')[2]
                assert mt in all_file_dfs.keys()
                mt_df = pd.read_csv(mt_file, header=None, index_col=False)
                mt_df.columns = ['file_ids', model]
                mt_df['file_ids'] = [os.path.basename(f) for f in mt_df['file_ids']]
                mt_df[model] = (mt_df[model] - np.mean(mt_df[model])) / np.std(mt_df[model])  # 分数归一化
                if np.isnan(mt_df[model]).any():
                    raise Exception(f'NaN in normalized score of {mt}')
                all_file_dfs[mt].append(mt_df)

        # 合并不同文件
        self.all_dfs = {'dev': {}, 'eval': {}}
        for mt in all_file_dfs.keys():
            mt_df = all_file_dfs[mt][0]
            for i in range(1, len(all_file_dfs[mt])):
                mt_df = pd.merge(mt_df, all_file_dfs[mt][i], on='file_ids', how='left')
            if mt in DEV_TYPES23:
                self.all_dfs['dev'][mt] = mt_df
            else:
                self.all_dfs['eval'][mt] = mt_df

    def search_best_coeff(self):
        if self.model_num == 1:
            self.best_coeff = [1]
            result, all_hm = self.cal_dev_hmean(self.best_coeff)
            self.coeff_hmean = {tuple([1]): {'result': result, 'all_hm': all_hm}}
            self.best_hm = all_hm
            print(f'best_hm: {all_hm * 100:.2f}; best_coeff: {self.best_coeff}')
        else:
            self.coeff_hmean = {}
            self.grid_result = []
            for i in np.arange(0, 1 + self.delta, self.delta):
                self.grid_search([i])
            self.grid_result = np.array(self.grid_result)
            self.best_hm = np.max(self.grid_result[:, 0])
            best_idx = np.argmax(self.grid_result[:, 0])
            self.best_coeff = self.grid_result[best_idx, 1:]
            print(f'best_hm: {self.best_hm * 100:.2f}; best_coeff: {self.best_coeff}')

            # 可视化
            if self.model_num == 2:
                self.visualize([0])
            elif self.model_num == 3:
                self.visualize([0, 1])  # 独立的维度

            # 写出
            dirn = 'csv_path/' + '_'.join(self.model)
            os.makedirs(dirn, exist_ok=True)
            coeff = self.best_coeff[np.newaxis, :].repeat(200, 0)
            for sn in ['dev', 'eval']:
                for mt in self.all_dfs[sn].keys():
                    df = self.all_dfs[sn][mt]
                    file_ids = df['file_ids']
                    scores = df[df.columns[1:]].to_numpy()
                    ensem_score = (scores * coeff).sum(1)
                    ndf = pd.DataFrame({'file_ids': file_ids, 'scores': ensem_score})
                    ndf.to_csv(os.path.join(dirn, f'anomaly_score_{mt}_section_0.csv'),
                               header=False, index=False)

    def grid_search(self, coeff):
        if len(coeff) == self.model_num - 1:
            coeff.append(1 - sum(coeff))
            result, all_hm = self.cal_dev_hmean(coeff)
            self.coeff_hmean[tuple(coeff)] = {'result': result, 'all_hm': all_hm}
            self.grid_result.append([all_hm] + coeff)
        else:
            for i in np.arange(0, 1 - sum(coeff) + self.delta, self.delta):
                self.grid_search(coeff + [i])

    def cal_dev_hmean(self, coeff):
        coeff = np.array(coeff)[np.newaxis, :]
        coeff = coeff.repeat(200, 0)
        result = []
        for mt in self.all_dfs['dev'].keys():
            df = self.all_dfs['dev'][mt]
            file_ids = df['file_ids'].tolist()
            status = np.array([1 if 'anomaly' in f else 0 for f in file_ids])
            scores = df[df.columns[1:]].to_numpy()
            ensem_score = (scores * coeff).sum(1)
            index_test_sou = [True if ('anomaly' in f or 'source' in f) else False for f in file_ids]
            index_test_tar = [True if ('anomaly' in f or 'target' in f) else False for f in file_ids]
            s_auc = roc_auc_score(status[index_test_sou], ensem_score[index_test_sou])
            t_auc = roc_auc_score(status[index_test_tar], ensem_score[index_test_tar])
            pauc = roc_auc_score(status, ensem_score, max_fpr=0.1)
            mt_hm = hmean([s_auc, t_auc, pauc])
            result.append([s_auc, t_auc, pauc, mt_hm])
        result = np.array(result)
        all_hm = hmean(result[:, -1])
        return result, all_hm

    def visualize(self, indices):
        if len(indices) == 1:
            coeff = np.arange(0, 1 + self.delta, self.delta)
            hms = self.grid_result[:, 0]
            fig, ax = plt.subplots()
            ax.plot(coeff, hms, marker='o')
            ax.set_xlabel(self.model[indices[0]])
            ax.set_ylabel('all_hmean')
            ax.set_title(f'best_hm: {self.best_hm * 100:.2f}\nbest_coeff: {self.best_coeff}')
            fig.savefig('csv_path/' + '_'.join(self.model) + '.png',
                        bbox_inches='tight')
        elif len(indices) == 2:
            x = self.grid_result[:, indices[0] + 1]
            y = self.grid_result[:, indices[1] + 1]
            z = self.grid_result[:, 0]
            fig = plt.figure()
            ax = Axes3D(fig)
            surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.1)
            fig.colorbar(surf, shrink=0.5, aspect=5)
            ax.set_xlabel(self.model[indices[0]])
            ax.set_ylabel(self.model[indices[1]])
            ax.set_zlabel('all_hmean')
            ax.set_title(f'best_hm: {self.best_hm * 100:.2f}\nbest_coeff: {self.best_coeff}')
            fig.savefig('csv_path/' + '_'.join(self.model) + '.png',
                        bbox_inches='tight')


if __name__ == '__main__':
    # score_dirs = {'mfn': 'csv_path/mfn/none/',
    #               'nfcdee': 'csv_path/nfcdee/all'}
    score_dirs = {'mfn': 'csv_path/mfn/none/',
                  'w2v': 'csv_path/wav2vec_300m_pool/none',
                  'nfcdee': 'csv_path/nfcdee/group/'}
    # score_dirs = {'nfcdee': 'csv_path/nfcdee/group'}
    delta = 0.1
    GS = GridSearch(score_dirs, delta)
    GS.search_best_coeff()

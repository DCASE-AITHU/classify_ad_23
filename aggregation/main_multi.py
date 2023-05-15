import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from kaldiio import ReadHelper

from pt_w2v_utils import Classifier, BatchNorm1d, AttentiveStatisticsPooling
from losses import LogSoftmaxWrapper, AdditiveAngularMargin

from dataset import Dataset

from asd_methods import select_asd
from util import ensemble_embs
from sklearn.metrics import roc_auc_score
from scipy.stats import hmean


def domsp_detect(backend, domsp, test_embs, train_embs, train_domains, test_dom_pred):
    if domsp == 'none':
        test_scores = select_asd(backend, test_embs, train_embs)
    else:
        tr_sou_id = np.where(train_domains == 0)[0]
        tr_tar_id = np.where(train_domains == 1)[0]
        test_scores_sou = select_asd(backend, test_embs, train_embs[tr_sou_id])
        test_scores_sou = (test_scores_sou - np.mean(test_scores_sou)) / np.std(test_scores_sou)
        test_scores_tar = select_asd(backend, test_embs, train_embs[tr_tar_id])
        test_scores_tar = (test_scores_tar - np.mean(test_scores_tar)) / np.std(test_scores_tar)
        test_scores_dom = np.vstack([test_scores_sou, test_scores_tar]).T  # 200 * 2
        if domsp == 'hard':
            thres_tar = 0.5
            dom_coeff = np.vstack([np.ones(test_scores_dom.shape[0]),
                                   np.zeros(test_scores_dom.shape[0])]).T
            pred_tar = np.where(test_dom_pred[:, 1] > thres_tar)[0]
            dom_coeff[pred_tar] = np.array([0, 1])
            test_scores = (test_scores_dom * dom_coeff).sum(axis=-1)
        elif domsp == 'soft':
            dom_coeff = test_dom_pred
            test_scores = (test_scores_dom * dom_coeff).sum(axis=-1)
        elif domsp == 'min':
            test_scores = test_scores_dom.min(axis=-1)
    return test_scores


def asd_score(model, scp, machine, backends, domsp):
    all_vectors_test = []
    all_vectors_train = []
    train_domains = []
    test_domains = []
    test_labels = []
    test_dom_pred = []
    model.eval()

    dom_pred_df = pd.read_csv('dom_pred/mfn_pred_mtdom.csv')
    dom_pred_df = dom_pred_df[dom_pred_df['mt'] == machine]

    with ReadHelper("scp:" + scp) as reader:
        for key, numpy_array in reader:
            if machine not in key:  # process one machine at one time
                continue  # 挑选出该机器的样本
            if 'train' in key:
                with torch.no_grad():
                    all_vectors_train.append(model(torch.tensor(numpy_array).cuda().unsqueeze(0)).squeeze(0).cpu().numpy())
                if 'source' in key:
                    train_domains.append(0)
                else:
                    train_domains.append(1)
            else:
                with torch.no_grad():
                    all_vectors_test.append(model(torch.tensor(numpy_array).cuda().unsqueeze(0)).squeeze(0).cpu().numpy())
                if 'anomaly' in key:
                    test_domains.append(-1)  # anomaly
                    test_labels.append(1)
                elif 'source' in key:
                    test_domains.append(0)  # source normal
                    test_labels.append(0)
                else:
                    test_domains.append(1)  # target normal
                    test_labels.append(0)
                if domsp in ['hard', 'soft']:
                    test_dom_pred.append(dom_pred_df[dom_pred_df['name'] == key[len(machine) + 1:] + '.wav'].to_numpy()[0, -2:])
    train_domains = np.array(train_domains)
    test_domains = np.array(test_domains)
    test_labels = np.array(test_labels)
    test_dom_pred = np.array(test_dom_pred)
    source_index = np.logical_or(test_domains == -1, test_domains == 0)
    target_index = np.logical_or(test_domains == -1, test_domains == 1)

    results = []
    for backend in backends:
        test_scores = domsp_detect(backend, domsp, np.stack(all_vectors_test, axis=0),
                                   np.stack(all_vectors_train, axis=0),
                                   train_domains, test_dom_pred)
        s_auc = roc_auc_score(test_labels[source_index], test_scores[source_index])
        t_auc = roc_auc_score(test_labels[target_index], test_scores[target_index])
        p_auc = roc_auc_score(test_labels, test_scores, max_fpr=0.1)
        hmean_auc = hmean([s_auc, t_auc, p_auc])
        results.append(hmean_auc)
        # print("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(backend, s_auc, t_auc, p_auc, hmean_auc))
    model.train()
    return results


def asd_score_models(models, scp, machine, backends, output_path, domsp, test_mode=False):
    # models: [model in each round]
    os.system('mkdir -p {}'.format(output_path))
    all_vectors_test = []
    all_vectors_train = []
    train_domains = []
    test_domains = []
    test_labels = []
    test_dom_pred = []
    test_keys = []

    dom_pred_df = pd.read_csv('dom_pred/mfn_pred_mtdom.csv')
    dom_pred_df = dom_pred_df[dom_pred_df['mt'] == machine]

    with ReadHelper("scp:" + scp) as reader:
        for key, numpy_array in reader:
            if machine not in key:  # process one machine at one time
                continue
            if 'train' in key:
                with torch.no_grad():
                    temp_vectors = []
                    for model in models:
                        # 将训练集embed分别送入每轮的model
                        temp_vectors.append(model(torch.tensor(numpy_array).cuda().unsqueeze(0)).squeeze(0).cpu().numpy())
                    all_vectors_train.append(np.hstack(temp_vectors))
                if 'source' in key:
                    train_domains.append(0)
                else:
                    train_domains.append(1)
            else:
                with torch.no_grad():
                    temp_vectors = []
                    for model in models:
                        temp_vectors.append(model(torch.tensor(numpy_array).cuda().unsqueeze(0)).squeeze(0).cpu().numpy())
                    all_vectors_test.append(np.hstack(temp_vectors))
                if 'anomaly' in key:
                    test_domains.append(-1)  # anomaly
                    test_labels.append(1)
                elif 'source' in key:
                    test_domains.append(0)  # source normal
                    test_labels.append(0)
                else:
                    test_domains.append(1)  # target normal
                    test_labels.append(0)
                test_dom_pred.append(dom_pred_df[dom_pred_df['name'] == key[len(machine) + 1:] + '.wav'].to_numpy()[0, -2:])
                test_keys.append(key)
    train_domains = np.array(train_domains)
    test_domains = np.array(test_domains)
    test_labels = np.array(test_labels)
    test_dom_pred = np.array(test_dom_pred)
    source_index = np.logical_or(test_domains == -1, test_domains == 0)
    target_index = np.logical_or(test_domains == -1, test_domains == 1)

    results = []
    full_re = []
    for backend in backends:
        test_scores = domsp_detect(backend, domsp, np.stack(all_vectors_test, axis=0),  # 200*384
                                   np.stack(all_vectors_train, axis=0),  # 1000*384
                                   train_domains, test_dom_pred)
        # write output anomaly scores
        assert (len(test_labels) == len(test_keys) and len(test_keys) == len(test_scores))
        with open(output_path + "/anomaly_score_" + machine + "_section_00_test.csv", "w") as fout:
            for i, key in enumerate(test_keys):
                fout.write("{},{}\n".format(key[len(machine) + 1:] + ".wav", test_scores[i]))
        if test_mode:
            continue
        s_auc = roc_auc_score(test_labels[source_index], test_scores[source_index])
        t_auc = roc_auc_score(test_labels[target_index], test_scores[target_index])
        p_auc = roc_auc_score(test_labels, test_scores, max_fpr=0.1)
        hmean_auc = hmean([s_auc, t_auc, p_auc])
        results.append(hmean_auc)
        full_re.append([s_auc, t_auc, p_auc, hmean_auc])
        # print("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(backend, s_auc, t_auc, p_auc, hmean_auc))
    return results, np.array(full_re)


def accuracy(logits, y):
    acc = torch.sum(torch.eq(torch.argmax(logits, -1), y).to(torch.float32)) / y.shape[0]
    return acc


class Model(nn.Module):
    def __init__(self, embedding_dim=128, output_dim=119, nhead=8, num_layers=1):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, dim_feedforward=2048, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # self.pooling_module = AttentiveStatisticsPooling(embedding_dim, attention_channels=embedding_dim//2)
        # self.pooling_module_bn = BatchNorm1d(input_size=embedding_dim * 2)
        # self.fc = torch.nn.Linear(embedding_dim * 2, embedding_dim)
        self.fc = torch.nn.Linear(embedding_dim, embedding_dim)
        self.pooling_module_bn = BatchNorm1d(input_size=embedding_dim)  # 对encoder输出做pool

        # 分类头
        self.linear = Classifier(input_size=embedding_dim, out_neurons=self.output_dim)
        # loss definition
        self.loss = LogSoftmaxWrapper(AdditiveAngularMargin(margin=margin, scale=scale))

    def embedding(self, input):
        output = self.encoder(input)
        # output = self.fc( self.pooling_module_bn( self.pooling_module(output.transpose(1,2)) ).transpose(1,2) )
        output = self.fc(self.pooling_module_bn(output.mean(axis=1))).unsqueeze(1)
        return output.squeeze(1)

    def forward(self, input, labels=None):
        output = self.embedding(input)
        if self.training:
            output = self.linear(output)
            loss = self.loss(output, labels.to(torch.long))
            return loss, output
        else:
            return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 'arks/wav2vec_300m_2seval_interval200_0_condition_step_3600_method_knn_mean_hmean_63.842.scp'
    # 'arks/hubert_large_2seval_interval200_1_condition_step_8600_method_knn_mean_hmean_61.743.scp'
    parser.add_argument('--ark_scp', type=str, default='arks/hubert_large_2seval_interval200_1_condition_step_8600_method_knn_mean_hmean_61.743.scp')
    parser.add_argument('--n_times', type=int, default=3)
    parser.add_argument('--result_path', type=str, default='csv_path/hubert_large_pool')
    parser.add_argument('--domsp', choices=['none', 'hard', 'soft', 'min'], default='none')
    opt = parser.parse_args()
    margin = 0.2
    scale = 30
    # if len(sys.argv) != 4:  # 使用格式
    #     print("Usage:{} ark_scp run_times result_path".format(sys.argv[0]))
    #     exit(0)
    scp_file = opt.ark_scp
    multi_cnt = int(opt.n_times)
    model_name = '_'.join(opt.result_path.split('/')[1].split('_')[:-1])
    result_path = os.path.join(opt.result_path, opt.domsp)
    os.makedirs(result_path, exist_ok=True)
    domsp = opt.domsp
    dataset = Dataset(scp_file, seq_len=3)  # 随机截取seq_len个连续embed
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=True)
    input_shapes = []
    for i in dataloader:
        input_shapes = i[0].shape  # embedding大小
        break
    print(input_shapes)  # 2048*4*128
    print(dataset.__class_cnt__())  # 总类别数

    backends = ['knn']

    n_epochs = 1
    models = []
    machines_from_scp = {}
    dev_machines = ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve']
    test_machines = ['bandsaw', 'grinder', 'shaker', 'ToyDrone', 'ToyNscale', 'ToyTank', 'Vacuum']
    with open(scp_file, "r") as fin:
        for line in fin.readlines():
            machines_from_scp[line.split("-")[0]] = 1
    machines = []
    machines_test = []
    for machine in machines_from_scp.keys():
        if machine in dev_machines:
            machines.append(machine)
        if machine in test_machines:
            machines_test.append(machine)
    print(machines)
    for try_id in range(multi_cnt):  # 跑多少次
        model = Model(embedding_dim=input_shapes[-1], output_dim=dataset.__class_cnt__())
        model.cuda()
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),  # 仅选取有梯度的
                                     lr=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
        for n_epoch in range(n_epochs):
            train_accs = []
            train_losses = []
            for batch in dataloader:
                optimizer.zero_grad()
                loss, output = model(batch[0].cuda(), batch[1].unsqueeze(1).cuda())
                train_losses.append(loss.item())
                loss.backward()
                optimizer.step()
                with torch.no_grad():  # 训练时acc不到0.2，只需要简单训一下
                    acc = accuracy(output, batch[1].unsqueeze(1).cuda())  # output是logits
                    train_accs.append(acc)
            train_loss = sum(train_losses) / len(train_losses)
            scheduler.step(train_loss)
            train_acc = sum(train_accs) / len(train_accs)
            print('epoch: {}, lr: {} train_loss: {:.4f} train_acc: {:.4f} '
                  .format(n_epoch, optimizer.param_groups[0]['lr'], train_loss, train_acc))
            all_hmeans = []
            for machine in machines:  # 每次只推理1种机器
                results = asd_score(model, scp_file, machine, backends, domsp)
                all_hmeans.append(results)
            for i in range(len(backends)):
                final_hmean = []
                for j in range(len(machines)):
                    final_hmean.append(all_hmeans[j][i])  # mt => backend
                print("all_hmean {} {:.4f}".format(backends[i], hmean(final_hmean)))
        model.eval()
        models.append(model)

    all_hmeans = []
    full_hm = None
    for machine in machines:  # dev测试集
        results, full_re = asd_score_models(models, scp_file, machine, backends, result_path, domsp)
        all_hmeans.append(results)
        if full_hm is None:
            full_hm = full_re
        else:
            full_hm = np.hstack([full_hm, full_re])
    full_hm_hm = hmean(full_hm, axis=1)
    full_hm = np.round(np.hstack([full_hm, full_hm_hm[np.newaxis, :]]) * 100, 2)  # 1*29
    df = pd.DataFrame({be: full_hm[i, :] for i, be in enumerate(backends)})
    df.to_csv(f'pool_infer_result/{model_name}_{domsp}.csv')
    for i in range(len(backends)):  # i遍历后端
        final_hmean = []
        for j in range(len(machines)):  # j遍历样本
            print("{} {} Hmean: {:0.3f}".format(machines[j], backends[i], all_hmeans[j][i]))
            final_hmean.append(all_hmeans[j][i])
        print("all_hmean_models {} {:.4f}".format(backends[i], hmean(final_hmean)))
    for machine in machines_test:  # eval测试集
        results = asd_score_models(models, scp_file, machine, backends, result_path, domsp, test_mode=True)

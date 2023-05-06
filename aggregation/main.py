import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader

from kaldiio import ReadHelper

from aggregation.pt_w2v_utils import Classifier, BatchNorm1d, AttentiveStatisticsPooling
from aggregation.losses import LogSoftmaxWrapper, AdditiveAngularMargin

from aggregation.dataset import Dataset

from aggregation.asd_methods import select_asd
from aggregation.util import ensemble_embs
from sklearn.metrics import roc_auc_score
from scipy.stats import hmean

def asd_score(model, scp, machine, backends):
    all_vectors_test = []
    all_vectors_train = []
    test_domains = []
    test_labels = []
    model.eval()

    with ReadHelper("scp:"+scp) as reader:
        for key, numpy_array in reader:
            if machine not in key: # process one machine at one time
                continue
            if 'train' in key:
                with torch.no_grad():
                    all_vectors_train.append(model(torch.tensor(numpy_array).cuda().unsqueeze(0)).squeeze(0).cpu().numpy())
            else:
                with torch.no_grad():
                    all_vectors_test.append(model(torch.tensor(numpy_array).cuda().unsqueeze(0)).squeeze(0).cpu().numpy())
                if 'anomaly' in key:
                    test_domains.append(-1) # anomaly
                    test_labels.append(1)
                elif 'source' in key:
                    test_domains.append(0) # source normal
                    test_labels.append(0)
                else:
                    test_domains.append(1) # targget normal
                    test_labels.append(0)
    test_domains = np.array(test_domains)
    test_labels = np.array(test_labels)
    source_index = np.logical_or(test_domains ==-1, test_domains == 0)
    target_index = np.logical_or(test_domains ==-1, test_domains == 1)

    results = []
    for backend in backends:
        test_scores = select_asd(backend, np.stack(all_vectors_test, axis=0),
                                        np.stack(all_vectors_train, axis=0))
        s_auc = roc_auc_score(test_labels[source_index], test_scores[source_index])
        t_auc = roc_auc_score(test_labels[target_index], test_scores[target_index])
        p_auc = roc_auc_score(test_labels, test_scores)
        hmean_auc = hmean([s_auc, t_auc, p_auc])
        results.append(hmean_auc)
        #print("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(backend, s_auc, t_auc, p_auc, hmean_auc))
    model.train()
    return results

def accuracy(logits, y):
     acc = torch.sum(torch.eq(torch.argmax(logits, -1), y).to(torch.float32)) / y.shape[0]
     return acc

class Model(nn.Module):
    def __init__(self, embedding_dim=128, output_dim = 119, nhead=8, num_layers=1):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, dim_feedforward=2048, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # self.pooling_module = AttentiveStatisticsPooling(embedding_dim, attention_channels=embedding_dim//2)
        # self.pooling_module_bn = BatchNorm1d(input_size=embedding_dim * 2)
        # self.fc = torch.nn.Linear(embedding_dim * 2, embedding_dim)
        self.fc = torch.nn.Linear(embedding_dim, embedding_dim)
        self.pooling_module_bn = BatchNorm1d(input_size=embedding_dim)

        self.linear = Classifier(input_size=embedding_dim, out_neurons=self.output_dim)
        ### loss definition
        self.loss = LogSoftmaxWrapper(AdditiveAngularMargin(margin=margin, scale=scale))
    
    def embedding(self, input):
        output = self.encoder(input)
        #output = self.fc( self.pooling_module_bn( self.pooling_module(output.transpose(1,2)) ).transpose(1,2) )
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

import sys
#id=sys.argv[1]
margin=0.2 #0.2 #0.2 #float(sys.argv[2])
scale=30 #30 #float(sys.argv[3])
scp_file=sys.argv[1] #"arks/w2v_300m_2sdev_interval100_step5100_dur2_shift0.5.scp"
dataset = Dataset(scp_file, seq_len=3)

dataloader = DataLoader(dataset, batch_size = 2048, shuffle=True)
model = Model(output_dim = dataset.__class_cnt__())
model.cuda()
print(dataset.__class_cnt__())

#backends = ['lof', 'maha', 'knn', 'cos']
backends = ['knn']
#backends = ['cos']

n_epochs = 1
optimizer = torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, model.parameters()),
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
        with torch.no_grad():
            acc = accuracy(output, batch[1].unsqueeze(1).cuda())
            train_accs.append(acc)
    train_loss = sum(train_losses) / len(train_losses)
    scheduler.step(train_loss)
    train_acc = sum(train_accs) / len(train_accs)
    print('epoch: {}, lr: {} train_loss: {:.4f} train_acc: {:.4f} '.
                       format(n_epoch, optimizer.param_groups[0]['lr'], train_loss, train_acc))
    
    all_hmeans = []
    machines_from_scp = {}
    dev_machines = ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve']
    with open(scp_file, "r") as fin:
        for line in fin.readlines():
            machines_from_scp[ line.split("-")[0] ] = 1
    machines = []
    for machine in machines_from_scp.keys():
        if machine in dev_machines:
            machines.append(machine)
    print(machines)
    #machines = ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve']
    
    for machine in machines:
        results = asd_score(model, scp_file, machine, backends)
        all_hmeans.append(results)
    for i in range(len(backends)):
        final_hmean = []
        for j in range(len(machines)):
            final_hmean.append(all_hmeans[j][i])
        print("all_hmean {} {:.4f}".format(backends[i], hmean(final_hmean)))
        

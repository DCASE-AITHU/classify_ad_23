import os
import torch
import argparse
from kaldiio import WriteHelper

from datasets.machineData1d23 import (
    DEV_TYPES23,
    EVAL_TYPES23
)

import torchaudio
import glob

from w2v import W2V
import os

def list_all(data_root, machines):
    file_lists = []
    for machine in machines:
        file_lists_train = glob.glob("{}/{}/train/*wav".format(data_root, machine))
        file_lists_test = glob.glob("{}/{}/test/*wav".format(data_root, machine))
        file_lists.extend(file_lists_train)
        file_lists.extend(file_lists_test)
    return file_lists

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '--dataset', type=int, default=0, required=True, help="0:dev 1:eval 2:dev+eval")
    parser.add_argument('-ckpt', '--checkpoint', type=str, required=True)
    parser.add_argument('-or', '--output_root', type=str, required=True)
    parser.add_argument('-duration', '--input_duration', type=float, required=True)
    parser.add_argument('-shift', '--input_shift', type=float, required=True)
    parser.add_argument('-prefix', '--output_prefix', type=str, required=True)
    parser.add_argument('-model', '--model_name', type=str, default="wav2vec_300m")

    args = parser.parse_args()
    
    os.system('mkdir -p {}'.format(args.output_root))

    # 准备数据部分
    sr = 16000
    input_samples = int(16000 * args.input_duration)
    input_shift = int(16000 * args.input_shift)

    # 准备网络部分，网络中包含损失函数，训练时网络返回loss，测试时网络返回embedding
    if args.dataset == 0:
        n_classes = 119
    elif args.dataset == 1:
        n_classes = 48
    else:
        n_classes = 167

    emb_size = 128

    net = W2V(embedding_dim=emb_size, output_dim = n_classes, model_name=args.model_name)
    ckpt = args.checkpoint
    net.load_state_dict(torch.load(ckpt))
    net.cuda()
    net.eval()

    if args.dataset == 0:
        data_root = "/home/public/dcase/dcase23/updated/dev_data"
        file_lists_all = list_all(data_root, DEV_TYPES23)
    elif args.dataset == 1:
        data_root = "/home/public/dcase/dcase23/updated/eval_data"
        file_lists_all = list_all(data_root, EVAL_TYPES23)
    else:
        data_root = "/home/public/dcase/dcase23/updated/dev_data"
        file_lists_dev = list_all(data_root, DEV_TYPES23)
        data_root = "/home/public/dcase/dcase23/updated/eval_data"
        file_lists_test = list_all(data_root, EVAL_TYPES23)
        file_lists_all = file_lists_dev + file_lists_test
    with WriteHelper('ark,scp:{}/{}.ark,{}/{}.scp'.format(args.output_root, args.output_prefix,
                                                          args.output_root, args.output_prefix)) as writer:
        for file_name in file_lists_all:
            machine_id = file_name.split("/")[7]
            waveform, _ = torchaudio.load(file_name)
            waveform_utt = []
            segment_samples = input_samples
            segment_samples_shift = input_shift
            end = segment_samples
            while end < waveform.shape[1]:
                waveform_utt.append(waveform[:, end - segment_samples : end])
                end += segment_samples_shift
            waveform_utt.append(waveform[:, -segment_samples :]) 
            waveform_utt = torch.cat(waveform_utt, dim=0).to("cuda:0")
            with torch.no_grad():
                output = net(waveform_utt)
                embedding = output['embedding'].cpu()
                writer(machine_id+"-"+file_name.split("/")[-1].replace(".wav", ""), embedding.numpy())

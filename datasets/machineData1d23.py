import os
import numpy as np
import pandas as pd
import torch.utils.data
import glob
import soundfile as sf
from tqdm import tqdm
import random
from typing import List, Optional
from . import (
    CLASS_MAP23,
    INVERSE_CLASS_MAP23,
    INVERSE_CLASS_MAP23_DEV,
    INVERSE_CLASS_MAP23_EVAL,
    ALL_TYPES23,
    DEV_TYPES23,
    EVAL_TYPES23,
)
CLASS_MAP = CLASS_MAP23
INVERSE_CLASS_MAP = INVERSE_CLASS_MAP23


class MCMDataSet1d23():
    def __init__(
            self,
            machine_types: Optional[List or int] = 0,
            input_samples=16384,
            hop_size=None,
            data_root='/home/public/dcase/dcase23',
            task='machine',  # or 'condition'
            sp=False
    ):
        assert input_samples is not None, "input_samples should not None."
        if machine_types == -1:
            machine_types = list(INVERSE_CLASS_MAP.keys())
        if machine_types == -2:
            machine_types = list(INVERSE_CLASS_MAP23_DEV.keys())
        if machine_types == -3:
            machine_types = list(INVERSE_CLASS_MAP23_EVAL.keys())
        if type(machine_types) is not list:
            machine_types = [machine_types]
        self.data_root = data_root
        self.input_samples = input_samples
        self.hop_size = input_samples if hop_size is None else hop_size
        self.task = task
        training_sets = []
        embedding_sets = []
        validation_sets = []
        classify_labs = 0
        for machine_type in machine_types:
            training_sets.append(
                MachineDataSet(
                    machine_type, classify_labs,
                    mode='training',
                    input_samples=input_samples,
                    hop_size=None,
                    data_root=data_root,
                    task=self.task,
                    sp=sp
                )
            )
            if self.task == 'machine':
                classify_labs += 1
            else:
                classify_labs += training_sets[-1].__get_classify_labs_cnt__()

        for machine_name in ALL_TYPES23:
            machine_type = CLASS_MAP23[machine_name]
        # if INVERSE_CLASS_MAP[machine_type] in DEV_TYPES23:
            validation_sets.append(
                MachineDataSet(
                    machine_type, -1,  # for validation there is no need to keep correct section var labels
                    mode='validation',
                    input_samples=input_samples,
                    hop_size=self.hop_size,
                    data_root=data_root,
                    task='machine',  # no need change task here
                    sp=sp
                )
            )
            embedding_sets.append(
                MachineDataSet(
                    machine_type, -1,
                    mode='training',
                    input_samples=input_samples,
                    hop_size=self.hop_size,
                    data_root=data_root,
                    task='machine',  # no need change task here
                    sp=sp
                )
            )
        training_set = torch.utils.data.ConcatDataset(training_sets)  # 拼接完后仍按原来顺序
        embedding_set = torch.utils.data.ConcatDataset(embedding_sets)
        validation_set = torch.utils.data.ConcatDataset(validation_sets)
        self.training_set = training_set
        self.embedding_set = embedding_set
        self.validation_set = validation_set
        self.n_classes = classify_labs
        # assert self.n_classes > 1

    def training_data_set(self):
        return self.training_set

    def embedding_data_set(self):
        return self.embedding_set

    def validation_data_set(self):
        return self.validation_set


class MachineDataSet(torch.utils.data.Dataset):
    def __init__(
            self,
            machine_type: int,
            classify_labs: int,
            mode='training',
            input_samples=256,
            hop_size=None,
            task='machine',
            data_root='/home/public/dcase/dcase23',
            sp=False
    ):

        assert mode in ['training', 'validation']
        self.task = task
        self.data_root = data_root
        self.classify_labs = classify_labs
        if self.task == 'condition':
            self.classify_labs_cnt = 0
            self.classify_labs_dict = {}
        self.machine_type = INVERSE_CLASS_MAP[machine_type]
        self.eval_test = (self.machine_type in EVAL_TYPES23) and (mode == 'validation')
        self.mode = mode
        self.input_samples = input_samples
        self.hop_size = hop_size
        if self.machine_type in DEV_TYPES23:
            if sp:
                root_folder = 'dev_data_sp'
            else:
                root_folder = 'dev_data'
        elif self.machine_type in EVAL_TYPES23:
            if sp:
                root_folder = 'eval_data_sp'
            else:
                root_folder = 'eval_data'
        else:
            raise AttributeError
        if mode == 'training':
            files = glob.glob(
                os.path.join(
                    data_root, root_folder,
                    self.machine_type,
                    'train', '*.wav'
                )
            )
        elif mode == 'validation':
            files = glob.glob(
                os.path.join(
                    data_root, root_folder,
                    self.machine_type,
                    'test', '*.wav'
                )
            )
        else:
            raise AttributeError

        assert len(files) > 0

        self.files = sorted(files)
        self.data = self.__load_data__(self.files)
        self.meta_data = self.__load_meta_data__(self.files)
        assert len(self.data) == len(self.files), 'len(self.data)!=len(self.files)'
        self.index_map = {}
        ctr = 0
        for i, x in enumerate(self.data):
            n_points = len(x)
            if self.hop_size:
                j = 0
                while j + self.input_samples <= n_points:
                    self.index_map[ctr] = (i, j)
                    ctr += 1
                    j += self.hop_size
                if j + self.input_samples != n_points and self.input_samples < n_points:
                    self.index_map[ctr] = (i, n_points - self.input_samples)
                    ctr += 1
            else:
                for j in random.sample(range(n_points - self.input_samples), n_points // self.input_samples):
                    self.index_map[ctr] = (i, j)
                    ctr += 1
        self.length = ctr

    def __get_classify_labs_cnt__(self):
        return self.classify_labs_cnt

    def __getitem__(self, item):
        file_idx, offset = self.index_map[item]
        observation = self.data[file_idx][offset: offset + self.input_samples].astype('float32')
        assert observation.shape[-1] == self.input_samples, 'observation.shape[-1]!=self.input_samples'
        meta_data = self.meta_data[file_idx].copy()
        meta_data['observations'] = observation
        return meta_data

    def __len__(self):
        return self.length

    def __load_meta_data__(self, files):
        data = []
        if self.mode == 'validation':
            self.dom_pred_df = pd.read_csv('dom_pred/mfn_pred_mtdom.csv')
            self.dom_pred_df = self.dom_pred_df[self.dom_pred_df['mt'] == self.machine_type]
        for f in files:
            md = self.__get_meta_data__(f)
            data.append(md)
        return data

    def __load_data__(self, files):
        data = []
        for f in tqdm(files):
            x, _ = sf.read(f)
            data.append(x)
        return data

    def __get_meta_data__(self, file_path):
        meta_data = os.path.split(file_path)[1].split('_')
        machine_type = CLASS_MAP[str.rsplit(file_path, '/', 3)[1]]
        assert self.machine_type == INVERSE_CLASS_MAP[machine_type]
        machine_section = int(meta_data[1])

        file_name = os.path.basename(file_path)
        if self.mode == 'validation':
            dom_pred = self.dom_pred_df[self.dom_pred_df['name'] == file_name].to_numpy()[0, -2:]
            dom_pred = np.array([v for v in dom_pred])
        else:
            dom_pred = np.array([-1, -1])

        if self.eval_test:
            return {
                'targets': -1,
                'machine_types': machine_type,
                'machine_sections': machine_section,
                'domains': -1,
                'classify_labs': -1,
                'file_ids': os.sep.join(
                    os.path.normpath(file_path).split(os.sep)[-4:]),
                'dom_pred': dom_pred
            }

        domain = meta_data[2]
        if self.task == 'condition':
            section_var = self.machine_type + "_" + "_".join(meta_data[6:])
            if not (section_var in self.classify_labs_dict):
                self.classify_labs_dict[section_var] = self.classify_labs
                self.classify_labs += 1
                self.classify_labs_cnt += 1

        if self.eval_test:
            pass
        elif 'normal' in meta_data:
            y = 0
        elif 'anomaly' in meta_data:
            y = 1
        else:
            raise AttributeError

        if self.task == 'machine':
            return {
                'targets': y,
                'machine_types': machine_type,
                'machine_sections': machine_section,
                'domains': domain,
                'classify_labs': self.classify_labs,
                'file_ids': os.sep.join(
                    os.path.normpath(file_path).split(os.sep)[-4:]),
                'dom_pred': dom_pred
            }
        else:
            return {
                'targets': y,
                'machine_types': machine_type,
                'machine_sections': machine_section,
                'domains': domain,
                'classify_labs': self.classify_labs_dict[section_var],
                'file_ids': os.sep.join(
                    os.path.normpath(file_path).split(os.sep)[-4:]),
                'dom_pred': dom_pred
            }

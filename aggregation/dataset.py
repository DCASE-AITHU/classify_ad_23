from kaldiio import ReadHelper
from torch.utils.data import Dataset
import random


class Dataset(Dataset):
    def __init__(self, scp, seq_len):
        super().__init__()
        self.scp = "scp:" + scp
        self.class_cnt = 0  # 类别总数
        self.class_map = {}
        self.data = []
        self.seq_len = seq_len
        with ReadHelper(self.scp) as reader:
            for key, numpy_array in reader:  # 每个numpy_array是该clip的所有segment的embedding
                if 'train' not in key:
                    continue
                machine_type = key.split("-")[0]
                condition_type = machine_type + "_" + "_".join(key.split("_")[6:])
                # condition_type = key.split("_")[0]
                if condition_type not in self.class_map:
                    self.class_map[condition_type] = self.class_cnt
                    self.class_cnt += 1
                self.data.append([self.class_map[condition_type],
                                  numpy_array, condition_type])

    def __class_cnt__(self):
        return self.class_cnt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        start = int(random.random() * 
                    (self.data[id][1].shape[0] - self.seq_len))
        return self.data[id][1][start: start + self.seq_len], self.data[id][0], self.data[id][2]


if __name__ == "__main__":
    dataset = Dataset("w2v_1s_condition_arks/w2v_1b_1s_valve.scp", 19)
    for i, _ in enumerate(dataset):
        print(i, _[0].shape, _[1], _[2])
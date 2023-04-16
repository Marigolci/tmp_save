import os.path

import torch
import numpy as np

class NWP_DataSet(torch.utils.data.Dataset):
    def str2list(self, cur_s):
        cur_list = cur_s.split(',')
        cur_list = list(map(int, cur_list))
        return cur_list

    def __init__(self, train_file, read_size=50000, max_len=20, is_pred=False):
        self.labels = []
        self.targets = []
        self.seqs = []
        self.is_last_file = False
        for i in range(read_size):
            line = train_file.readline()
            if not line:
                self.is_last_file = True
                break
            label, target, seq = line.strip().split("\t")
            self.labels.append(self.str2list(label))
            self.targets.append(self.str2list(target))
            self.seqs.append(self.str2list(seq))
        self.max_len = max_len
        self.is_pred = is_pred

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        cur_seq = self.seqs[idx]
        cur_target = self.targets[idx]
        cur_label = self.labels[idx]
        if not self.is_pred:
            cur_label = cur_label.index(1)
        if len(cur_seq) > self.max_len:
            cur_seq = cur_seq[:self.max_len]
        else:
            cur_seq = [0] * (self.max_len - len(cur_seq)) + cur_seq    # padding
        # cur_seq = [cur_seq] * len(cur_target)
        return torch.LongTensor(cur_seq), torch.LongTensor(cur_target), cur_label




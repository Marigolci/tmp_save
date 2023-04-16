import random

import pandas as pd
import os
from tqdm import tqdm
from collections import defaultdict

class DataGenerationForRank(object):
    def __init__(self, root_path='../../', is_eval=True, **kwargs):
        self.root_path = root_path
        self.is_eval = is_eval
        self.word2idx = defaultdict(int)
        self.idx2word = defaultdict(str)
        self.sample_idx = 0
        self.input_train_filename = kwargs['input_train_filename']
        self.output_train_filename = kwargs['output_train_filename']
        self.input_test_filename = kwargs['input_test_filename']
        self.output_test_filename = kwargs['output_test_filename']
        self.word_dict_filename = kwargs['word_dict_filename']
        self.neg_size = kwargs['neg_sample_size']


    def gen_pos_data(self, sents, labels):
        pos_ins = {"prev_items": [], 'next_item': []}
        for prev_item, next_item in tqdm(zip(sents, labels)):
            if len(prev_item) > 1:
                for i in range(len(prev_item)-1):
                    pos_ins['prev_items'].append(prev_item[:i+1])
                    pos_ins['next_item'].append(prev_item[i+1])
        sents += pos_ins['prev_items']
        labels += pos_ins['next_item']
        return sents, labels

    def add_word(self, item):
        if self.word2idx[item] == 0:
            self.word2idx[item] = len(self.word2idx)
            self.idx2word[len(self.word2idx)] = item

    def gen_idx(self, seq):
        seq_idx = []
        for item in seq:
            self.add_word(item)
            seq_idx.append(self.word2idx[item])
        return seq_idx

    def gen_seq_and_label(self, df, has_label=True):
        sents = df['prev_items'].tolist()
        sents = list(map(eval, sents))
        if has_label:
            targets = df['next_item'].tolist()
            return sents, targets
        else:
            targets = df['next_item_prediction'].tolist()
            targets = list(map(eval, targets))
            return sents, targets

    def save_word_idx(self, dict_filename):
        dict_file = open(dict_filename, 'w')
        for key in self.word2idx:
            dict_file.write("{}:{}\n".format(key, self.word2idx[key]))

    def add_sample_idx(self):
        self.sample_idx += 1
        if self.sample_idx >= len(self.pooling):
            self.sample_idx = 0
            random.shuffle(self.pooling)

    def gen_neg_list(self, t, sample_size=20):
        res = [t]
        for i in range(sample_size):
            while self.pooling[self.sample_idx] in res:
                self.add_sample_idx()
            res.append(self.pooling[self.sample_idx])
            self.add_sample_idx()
        return res

    def save_idx(self, output_filename, sents, target, candidates, is_for_test=False):
        output_file = open(output_filename, 'w')
        for s,t,c in tqdm(zip(sents, target, candidates)):
            s = list(map(str, s))
            if not is_for_test:
                c = [item for item in c if item != t]
                t = self.gen_neg_list(t, sample_size=self.neg_size - len(c))
                t = t + c
            t = list(map(str, t))
            l = [0] * len(t)
            if not is_for_test:
                l[0] = 1
            l = list(map(str, l))
            output_file.write("{}\t{}\t{}\n".format(",".join(l), ",".join(t), ",".join(s)))

    def read_train_data(self, path_name):
        train_sents = []
        train_target = []
        train_candidate = []
        for line in tqdm(open(path_name, 'r')):
            line = line.strip()
            target, sents, candidates = line.split("\t")
            sents, candidates = sents.split(","), candidates.split(",")
            train_sents.append(sents)
            train_target.append(target)
            train_candidate.append(candidates)
        return train_sents, train_target, train_candidate

    def process(self):
        print("read train data")
        train_sents, train_target, candidates = self.read_train_data(
            os.path.join(self.root_path, 'input_data', self.input_train_filename))
        df_test = pd.read_csv(os.path.join(self.root_path, 'sort_data', self.input_test_filename))
        print("gen idx")
        train_sents = [self.gen_idx(s) for s in train_sents]
        candidates = [self.gen_idx(c) for c in candidates]
        train_target = self.gen_idx(train_target)
        self.pooling = train_target.copy()
        random.shuffle(self.pooling)
        output_train_filename = os.path.join(self.root_path, 'input_data', self.output_train_filename)
        output_test_filename = os.path.join(self.root_path, 'sort_data', self.output_test_filename)
        dict_filename = os.path.join(self.root_path, 'input_data', self.word_dict_filename)
        print("save train idx")
        self.save_idx(output_train_filename, train_sents, train_target, candidates)
        test_sents, test_targets = self.gen_seq_and_label(df_test, has_label=False)
        print("gen test idx")
        test_sents = [self.gen_idx(s) for s in test_sents]
        test_targets = [self.gen_idx(s) for s in test_targets]
        print("save test idx")
        self.save_idx(output_test_filename, test_sents, test_targets, is_for_test=True)
        self.save_word_idx(dict_filename)

if __name__ == "__main__":
    d = DataGenerationForRank()
    d.process()
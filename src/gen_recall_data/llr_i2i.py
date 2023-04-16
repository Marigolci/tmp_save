import math
import os.path
import pickle
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


class LLRI2I(object):
    def __init__(self, is_eval=True, root_path="../../"):
        self.root_path = root_path
        self.is_eval = is_eval
        self.item_pair_dict = defaultdict(lambda: defaultdict(float))
        self.item_count_dict = defaultdict(float)
        if is_eval:
            self.result_file = 'recall_data/LLRI2I_for_eval.csv'
        else:
            self.result_file = 'recall_data/LLRI2I.csv'

    def process_data(self, df, has_label=False):
        for _, row in tqdm(df.iterrows()):
            seq = eval(row['prev_items'])
            if has_label:
                seq += [row['next_item']]
            if len(seq) <= 1:
                continue
            for i in range(len(seq)-1):
                for j in range(i+1, len(seq)):
                    self.item_pair_dict[seq[i]][seq[j]] += 1.0 / (j-i)
                    self.item_count_dict[seq[i]] += 1.0 / (j-i)
                    self.item_count_dict[seq[j]] += 1.0 / (j-i)

    def entropy(self, *args):
        sum_element = sum(args)
        result = 0.0
        for k in args:
            if k < 0:
                result += 0
            result += k * math.log((k + 1)/sum_element)
        return -result

    def gen_scores(self, N=100000):
        for item_i in self.item_pair_dict:
            for item_j in self.item_pair_dict[item_i]:
                K11 = self.item_pair_dict[item_i][item_j]
                K12 = self.item_count_dict[item_i] - K11
                K21 = self.item_count_dict[item_j] - K11
                K22 = N - K12 - K21 + K11
                # print("K11 is:{}, K12 is:{}, K21 is:{}, K22 is:{}, score is:{}, item i is:{}, item j is:{}".format(K11, K12, K21, K22, self.item_pair_dict[item_i][item_j], self.item_count_dict[item_i], self.item_count_dict[item_j]))
                self.item_pair_dict[item_i][item_j] = self.entropy(K11, K12, K21, K22) - self.entropy(K11, K12) - self.entropy(K21, K22) - self.entropy(K11, K21) - self.entropy(K12, K22)



    def save_data(self, data):
        f = open(os.path.join(self.root_path, self.result_file), 'w')
        f.write('trigger_name,recall_item,recall_weight\n')
        for trigger in data:
            for item in data[trigger]:
                f.write("{},{},{}\n".format(trigger, item, data[trigger][item]))
        f.close()

    def process(self):
        df_train = pd.read_csv(os.path.join(self.root_path, 'input_data/train.csv'))
        df_eval = pd.read_csv(os.path.join(self.root_path,'input_data/eval.csv'))
        df_test = pd.read_csv(os.path.join(self.root_path,'input_data/test.csv'))
        print("process train data in LLRI2I")
        self.process_data(df_train, True)
        self.gen_scores()
        if self.is_eval:
            print("process eval data in LLRI2I")
            self.process_data(df_eval, False)
        else:
            print("process eval data in LLRI2I")
            self.process_data(df_eval, True)
            print("process test data in LLRI2I")
            self.process_data(df_test, False)
        self.save_data(self.item_pair_dict)


if __name__ == '__main__':
    d = LLRI2I()
    d.process()
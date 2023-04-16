import pickle

import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import os


class Dhot(object):
    def __init__(self, is_eval=True, root_path="/home/lpf/lpf/rec/"):
        self.is_eval = is_eval
        self.root_path = root_path
        self.hot_item_dict = defaultdict(lambda: defaultdict(int))
        if self.is_eval:
            self.result_file = 'recall_data/dhot_for_eval.csv'
        else:
            self.result_file = 'recall_data/dhot.csv'

    def process_data(self, df, has_label=False):
        for _, row in tqdm(df.iterrows()):
            seq = eval(row['prev_items'])
            if has_label:
                seq += [row['next_item']]
            distinct = row['locale']
            for item in seq:
                self.hot_item_dict[distinct][item] += 1

    def save_data(self, data):
        f = open(os.path.join(self.root_path, self.result_file), 'w')
        f.write('loc,hot_item,recall_weight\n')
        for trigger in data:
            for item in data[trigger]:
                f.write("{},{},{}\n".format(trigger, item, data[trigger][item]))
        f.close()

    def process(self):
        df_train = pd.read_csv(os.path.join(self.root_path, 'input_data/train.csv'))
        df_eval = pd.read_csv(os.path.join(self.root_path,'input_data/eval.csv'))
        df_test = pd.read_csv(os.path.join(self.root_path,'input_data/test.csv'))
        print("process train data in hoti2i")
        self.process_data(df_train, True)
        if self.is_eval:
            print("process eval data in hoti2i")
            self.process_data(df_eval, False)
        else:
            print("process eval data in hoti2i")
            self.process_data(df_eval, True)
            print("process test data in hoti2i")
            self.process_data(df_test, False)
        self.save_data(self.hot_item_dict)

if __name__ == '__main__':
    d = Dhot(is_eval=False)
    d.process()
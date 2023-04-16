import os.path
import pickle
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


class AdaI2I(object):
    def __init__(self, is_eval=True, root_path="../../", is_for_train=False, top=10):
        self.root_path = root_path
        self.is_eval = is_eval
        self.item_pair_dict = defaultdict(lambda: defaultdict(float))
        if is_eval:
            self.result_file = 'recall_data/adai2i_for_eval.csv'
        else:
            self.result_file = 'recall_data/adai2i.csv'
        self.is_for_train = is_for_train
        if is_for_train:
            self.result_file = 'recall_data/adai2i_for_train.csv'
        self.top = top

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

    def gen_recall_data(self, df):
        prev_items = df['prev_items'].tolist()
        prev_items = [eval(p) for p in prev_items]
        pred_list = []
        for prev_item in prev_items:
            pred = defaultdict(float)
            for i, trigger in enumerate(prev_item):
                for recall_item in self.item_pair_dict[trigger]:
                    pred[recall_item] = self.item_pair_dict[trigger][recall_item] / ((len(prev_item) - i) ** 2)
            pred = sorted(pred.items(), key=lambda k:k[1], reverse=True)[:self.top]
            pred = list(map(lambda k:k[0], pred))
            pred_list.append(pred)
        df['ada_i2i_recall'] = pred_list
        df.to_csv(os.path.join(self.root_path, self.result_file), index=False)

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
        print("process train data in adai2i")
        if self.is_for_train:
            self.process_data(df_train, False)
        else:
            self.process_data(df_train, True)
            if self.is_eval:
                print("process eval data in adai2i")
                self.process_data(df_eval, False)
            else:
                print("process eval data in adai2i")
                self.process_data(df_eval, True)
                print("process test data in adai2i")
                self.process_data(df_test, False)
            self.save_data(self.item_pair_dict)


if __name__ == '__main__':
    d = AdaI2I()
    d.process()
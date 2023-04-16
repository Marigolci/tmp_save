import os.path
import pickle
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


class TxtI2I(object):
    def __init__(self, is_eval=True, root_path="../../"):
        self.root_path = root_path
        self.is_eval = is_eval
        self.item_pair_dict = defaultdict(lambda: defaultdict(float))
        self.sim_path = os.path.join(root_path, "recall_data/txt_sim_scores_roberta_large_title_top100_new.txt")

        if is_eval:
            self.src_file = 'recall_data/adai2i_for_eval.csv'
            self.result_file = "recall_data/adai2i_txt_for_eval.csv"
        else:
            self.src_file = 'recall_data/adai2i.csv'
            self.result_file = "recall_data/adai2i_txt.csv"

        self.src_file = os.path.join(root_path, self.src_file)
        self.result_file = os.path.join(root_path, self.result_file)
        print(self.src_file)

    def process(self):
        fd = open(self.sim_path,'r')
        lines = fd.readlines()
        fd.close()
        results = defaultdict(dict)
        lines = lines[1:]
        for line in tqdm(lines):
            line_list = line.strip().split(',')
            item1 = line_list[0]
            item2 = line_list[1]
            score = float(line_list[2])
            results[item1][item2] = score
        df = pd.read_csv(self.src_file)
        results_new = defaultdict(list)
        for idx, row in tqdm(df.iterrows(), ncols = len(df)):
            item1 = row['trigger_name']
            item2 = row['recall_item']
            # weight = row['recall_weight']
            if item1 in results and item2 in results[item1]:
                results_new['trigger_name'].append(item1)
                results_new['recall_item'].append(item2)
                results_new['recall_weight'].append(results[item1][item2])
        self.save_data(results_new)


    def save_data(self, data):
        pd_res = pd.DataFrame(data)
        pd_res.to_csv(self.result_file, index=False)


if __name__ == '__main__':
    d = AdaI2I()
    d.process()
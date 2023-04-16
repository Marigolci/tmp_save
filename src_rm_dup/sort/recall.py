import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import os

class RecallModule(object):
    def __init__(self, is_eval=True, root_path='/home/lpf/lpf/rec/'):
        self.is_eval=is_eval
        self.root_path = root_path
        if self.is_eval:
            self.ada_i2i = open(os.path.join(root_path, 'recall_data/adai2i_for_eval.csv'), 'r')
            self.dhot = pd.read_csv(os.path.join(root_path, 'recall_data/dhot_for_eval.csv'))
        else:
            self.ada_i2i = open(os.path.join(root_path, 'recall_data/adai2i.csv'), 'r')
            self.dhot = pd.read_csv(os.path.join(root_path, 'recall_data/dhot.csv'))
        self.preprocess_i2i_data()
        self.hot_cache = {}

    def preprocess_i2i_data(self):
        # self.i2i_result = defaultdict(list)
        self.i2i_dict = defaultdict(lambda: defaultdict(float))
        print("read i2i data in recall module")
        isHead=True
        for line in tqdm(self.ada_i2i):
            if isHead:
                isHead = False
                continue
            ele = line.strip().split(',')
            self.i2i_dict[ele[0]][ele[1]] = float(ele[2])
        # for i, row in tqdm(self.ada_i2i.iterrows()):
        #     i2i_dict[row['trigger_name']][row['recall_item']] = row['recall_weight']
        # for key in tqdm(i2i_dict):
        #     recall_item_dict = i2i_dict[key]
        #     recall_item = sorted(recall_item_dict.items(), key=lambda k:k[1], reverse=True)
        #     recall_item = list(map(lambda k:k[0], recall_item))
        #     self.i2i_result[key] = recall_item


    def get_hot_result(self, loc):
        if loc not in self.hot_cache:
            hot_result = self.dhot[self.dhot['loc'] == loc].sort_values(by='recall_weight', axis=0, ascending=False)[
                         :200]
            hot_item = hot_result['hot_item'].to_list()
            recall_weight = hot_result['recall_weight'].to_list()
            self.hot_cache[loc] = dict(zip(hot_item, recall_weight))
        return self.hot_cache[loc]


    def process(self):
        if self.is_eval:
            df_test = pd.read_csv(os.path.join(self.root_path, 'input_data/eval.csv'))
        else:
            df_test = pd.read_csv(os.path.join(self.root_path, 'input_data/test.csv'))
        preds = []
        df_test['last_item'] = df_test['prev_items'].apply(lambda x: eval(x)[-1])
        # df_test['next_item_prediction'] = df_test['last_item'].map(self.i2i_result)
        i2i_hit_cnt = 0
        print("handle data in recall module")
        count = 0
        for _, row in tqdm(df_test.iterrows()):
            loc = row['locale']
            trigger = row['last_item']
            hot_result = self.get_hot_result(loc)
            i2i_result_dict = self.i2i_dict[trigger]
            prev_items = eval(row['prev_items'])
            if i2i_result_dict is not None and len(i2i_result_dict) > 0:
                i2i_hit_cnt += 1
            pred = defaultdict(float)
            for item in i2i_result_dict:
                if item not in prev_items: 
                    pred[item] += i2i_result_dict[item] * 10000
                if item in prev_items:
                    count+=1
                    
            for item in hot_result:
                if item not in prev_items:
                    pred[item] += hot_result[item] * 0.00001
                    
            
                    
            pred = sorted(pred.items(), key=lambda k:k[1], reverse=True)[:100]
            pred = list(map(lambda k:k[0], pred))
            preds.append(pred)
        
        print("i2i_hit_ratio is:", i2i_hit_cnt/len(df_test))
        df_test['next_item_prediction'] = preds
        if self.is_eval:
            df_test.to_csv(os.path.join(self.root_path, 'sort_data/recall_result_for_eval_0.csv'), index=False)
        else:
            df_test.to_csv(os.path.join(self.root_path, 'sort_data/recall_result.csv'), index=False)

if __name__ == '__main__':
    d = RecallModule(is_eval=False)
    d.process()
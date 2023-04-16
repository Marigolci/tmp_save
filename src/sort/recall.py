import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import os

class RecallModule(object):
    def __init__(self, is_eval=True, root_path='../../', **kwargs):
        self.is_eval=is_eval
        self.root_path = root_path
        self.recall_param = kwargs['params']
        if self.recall_param['ada_i2i']['enable']:
            self.ada_i2i = open(os.path.join(root_path, 'recall_data', kwargs['ada_i2i']), 'r')
            self.i2i_dict = self.preprocess_i2i_data(self.ada_i2i)
        if self.recall_param['ada_i2i_txt']['enable']:
            self.ada_i2i_txt = open(os.path.join(root_path, 'recall_data', kwargs['ada_i2i_txt']), 'r')
            self.i2i_txt_dict = self.preprocess_i2i_data(self.ada_i2i_txt)
        if self.recall_param['dhot']['enable']:
            self.dhot = pd.read_csv(os.path.join(root_path, 'recall_data', kwargs['dhot']))
        if self.recall_param['fasttext']['enable']:
            self.fasttext = pd.read_csv(os.path.join(root_path, 'recall_data', kwargs['fasttext']))
            self.fasttext_item = self.fasttext['next_item_prediction'].tolist()
            self.fasttext_scores = self.fasttext['fasttext_scores'].tolist()
        # self.llr_dict = self.preprocess_i2i_data(self.llr_i2i, 'llr')
        self.hot_cache = {}
        self.id2author = defaultdict(dict)
        self.product_path = os.path.join(root_path, "data", kwargs['product_path'])
        self.gen_prodcut_info()
        if self.recall_param['title_sim']['enable']:
            self.title_sim_path = os.path.join(root_path, "recall_data/item_sim_title_0.00078.csv")
            self.id2sims = defaultdict(set)
            self.gen_sim_titles()
        self.df_test = pd.read_csv(os.path.join(self.root_path, 'input_data', kwargs['input_filename']))
        self.dst_filename = os.path.join(self.root_path, 'sort_data', kwargs['dst_filename'])


    def gen_sim_titles(self):
        print("gen sim titles")
        df = pd.read_csv(self.title_sim_path)
        for idx, row in tqdm(df.iterrows()):
            query_item = row['query_item']
            sim_items = eval(row['sim_items'])
            for item in sim_items:
                self.id2sims[query_item].add(item)

    def gen_prodcut_info(self):
        print("gen product info")
        df = pd.read_csv(self.product_path)
        for idx,row in tqdm(df.iterrows()):
            id = row['id']
            author = row['author']
            locale = row["locale"]
            self.id2author[locale][id] = author

    def preprocess_i2i_data(self, input_file, name='ada'):
        # self.i2i_result = defaultdict(list)
        i2i_dict = defaultdict(lambda: defaultdict(float))
        print("read {} i2i data in recall module".format(name))
        isHead=True
        for line in tqdm(input_file):
            if isHead:
                isHead = False
                continue
            ele = line.strip().split(',')
            i2i_dict[ele[0]][ele[1]] = float(ele[2])
        return i2i_dict

    def get_hot_result(self, loc):
        if loc not in self.hot_cache:
            hot_result = self.dhot[self.dhot['loc'] == loc].sort_values(by='recall_weight', axis=0, ascending=False)[
                         :200]
            hot_item = hot_result['hot_item'].to_list()
            recall_weight = hot_result['recall_weight'].to_list()
            self.hot_cache[loc] = dict(zip(hot_item, recall_weight))
        return self.hot_cache[loc]


    def process(self):
        preds = []
        self.df_test['last_item'] = self.df_test['prev_items'].apply(lambda x: eval(x)[-1])
        i2i_hit_cnt = 0
        print("handle data in recall module")
        for i , row in tqdm(self.df_test.iterrows()):
            loc = row['locale']
            trigger = row['last_item']
            hot_result = self.get_hot_result(loc)
            # i2i_result_dict = self.i2i_dict[trigger]
            prev_items = eval(row['prev_items'])
            # prev_authors = set([self.id2author[loc][item.split('_')[1]] for item in prev_items])
#             if i2i_result_dict is not None and len(i2i_result_dict) > 0:
#                 i2i_hit_cnt += 1
            pred = defaultdict(float)
            if self.recall_param['fasttext']['enable']:
                for fasttext_item, fasttext_score in zip(eval(self.fasttext_item[i]), eval(self.fasttext_scores[i])):
                    if fasttext_item not in prev_items:
                        pred[fasttext_item] = fasttext_score * 30
            if self.recall_param['title_sim']['enable']:
                if trigger in self.id2sims:
                    for  item in self.id2sims[trigger]:
                        if item not in prev_items:
                            pred[item] += 1.
            for index,trig in  enumerate(prev_items):
                # only the last item
                if index < len(prev_items)-1 :
                    continue
                local_prev_items = prev_items[:index+1]
                prev_authors = set([self.id2author[loc][item.split('_')[1]] for item in local_prev_items])

                # llr_result_dict = self.llr_dict[trig]
                if self.recall_param['ada_i2i']['enable']:
                    i2i_result_dict = self.i2i_dict[trig]
                    for item in i2i_result_dict:
                        if self.id2author[loc][item.split('_')[1]] in prev_authors:
                            ratio = 2.
                        else:
                            ratio = 1.
                        if item not in prev_items:
                            pred[item] += ratio*i2i_result_dict[item] * 10 / ((len(prev_items) - index)**2)
                if self.recall_param['ada_i2i_txt']['enable']:
                    i2i_txt_result_dict = self.i2i_txt_dict[trig]
                    for item in i2i_txt_result_dict:
                        if self.id2author[loc][item.split('_')[1]] in prev_authors:
                            ratio = 2.
                        else:
                            ratio = 1.
                        if item not in prev_items:
                            pred[item] += ratio*i2i_txt_result_dict[item] * 10 / ((len(prev_items) - index)**2)
            if len(pred) > 0:
                i2i_hit_cnt += 1
            for item in hot_result:
                if item not in prev_items:
                    pred[item] += hot_result[item] * 0.00001
            pred = sorted(pred.items(), key=lambda k:k[1], reverse=True)[:100]
            pred = list(map(lambda k:k[0], pred))
            pred = [elem.split('_')[1] for elem in pred]
            preds.append(pred)
        print("i2i_hit_ratio is:", i2i_hit_cnt/len(self.df_test))
        self.df_test['next_item_prediction'] = preds
        self.df_test.to_csv(self.dst_filename, index=False)

if __name__ == '__main__':
    d = RecallModule()
    d.process()
import pandas as pd
from collections import defaultdict, Counter
import math
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os

def str2list(x):
    x = x.replace('[', '').replace(']', '').replace("'", '').replace('\n', ' ').replace('\r', ' ')
    l = [i for i in x.split() if i]
    return l

def concat_locale(df):
    for idx, row in tqdm(df.iterrows()):
        prev_items = row['prev_items']
        locale = row['locale']
        prev_items = [locale + '_' + item for item in prev_items]
        row['prev_items'] = prev_items

class DataGeneration(object):
    def __init__(self, ratio=0.1, root_path='../../'):
        self.eval_ratio = ratio
        self.root_path = root_path

    def process(self):
        df_train = pd.read_csv(os.path.join(self.root_path, 'data/sessions_train.csv'))
        df_test = pd.read_csv(os.path.join(self.root_path, 'data/sessions_test_task1.csv'))
        df_train['prev_items'] = df_train['prev_items'].apply(str2list)
        df_test['prev_items'] = df_test['prev_items'].apply(str2list)
        df_train['next_item'] = df_train[['locale','next_item']].apply(lambda x: '_'.join(x),axis=1)
        concat_locale(df_test)
        concat_locale(df_train)

        df_train, df_eval = train_test_split(df_train, test_size=self.eval_ratio, random_state=1234)
        df_train.to_csv(os.path.join(self.root_path, 'input_data/train.csv'), index=False)
        df_eval.to_csv(os.path.join(self.root_path, 'input_data/eval.csv'), index=False)
        df_test.to_csv(os.path.join(self.root_path, 'input_data/test.csv'), index=False)
        # print(df_train.head())

if __name__ == '__main__':
    d = DataGeneration()
    # d.process()
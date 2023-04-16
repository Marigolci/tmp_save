import pandas as pd
from tqdm import tqdm
import os

class Mrr(object):
    def __init__(self, root_path='/home/lpf/lpf/rec/'):
        self.root_path = root_path

    def process(self):
        df = pd.read_csv(os.path.join(self.root_path, 'sort_data/recall_result_for_eval_0.csv'))
        scores = 0.0
        for _,row in tqdm(df.iterrows()):
            label = row['next_item']
            pred = eval(row['next_item_prediction'])
            if label in pred:
                position = pred.index(label)
                scores += 1/(position+1)
        print("mmr result is:", scores / len(df))

if __name__ == '__main__':
    d = Mrr()
    d.process()


import pandas as pd
from tqdm import tqdm
import os

class Mrr(object):
    def __init__(self, root_path='../../', eval_file_name='sort_data/recall_result_for_eval.csv'):
        self.root_path = root_path
        self.eval_file = os.path.join(self.root_path, eval_file_name)

    def process(self):
        df = pd.read_csv(self.eval_file)
        scores = 0.0
        for _,row in tqdm(df.iterrows()):
            label = row['next_item'].split('_')[1]
            pred = eval(row['next_item_prediction'])
            if label in pred:
                position = pred.index(label)
                scores += 1/(position+1)
        print("mmr result is:", scores / len(df))

if __name__ == '__main__':
    d = Mrr(eval_file_name='sort_data/recall_result_for_eval.csv')
    d.process()


import pandas as pd
import os

class GenSubmission(object):
    def __init__(self, root_path='../../', submit_file_name='sort_data/recall_result.csv'):
        self.root_path = root_path
        self.fpath = os.path.join(self.root_path, submit_file_name)

    def process(self):
        df = pd.read_csv(self.fpath) 
        df['next_item_prediction'] = df['next_item_prediction'].apply(lambda r:eval(r))
        df[['locale', 'next_item_prediction']].to_parquet(os.path.join(self.root_path, 'sort_data/submission_task1.parquet'), engine='pyarrow')


if __name__ == '__main__':
    g = GenSubmission()
    g.process()
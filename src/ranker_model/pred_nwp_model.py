from nwp_dataset import NWP_DataSet
from simple_nwp_model import SimpleNWPModel
import os.path
import pickle
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
import torch.nn as nn
import torch
from tqdm import tqdm


def rank_item(row):
    origin_item = eval(row['next_item_prediction'])
    scores = row['rank_scores']
    res_item = sorted(list(zip(origin_item, scores)), key=lambda k:k[1], reverse=True)
    res_item = list(map(lambda k:k[0], res_item))
    return res_item


class PredNWPModel(object):
    def __init__(self, root_path='../../', is_eval=True, batch_size=16, model_name='simple_nwp_for_eval.pth'):
        self.root_path = root_path
        self.is_eval = is_eval
        if is_eval:
            self.pred_filename = os.path.join(self.root_path, 'sort_data/rank_prepare_for_eval.bin')
        else:
            self.pred_filename = os.path.join(self.root_path, 'sort_data/rank_prepare.bin')
        self.model_name = model_name
        self.batch_size = batch_size

    def process(self):
        pred_file = open(self.pred_filename, 'r')
        pred_dataset = NWP_DataSet(pred_file, read_size=100*10000, is_pred=True)
        pred_dataloader = torch.utils.data.DataLoader(pred_dataset, shuffle=False, batch_size=self.batch_size)
        model = torch.load(os.path.join(self.root_path, 'rank_data', self.model_name))
        model.eval()
        prediction_scores = []
        for batch in tqdm(pred_dataloader):
            seq, target, label = batch
            label_size = target.shape[1]
            target = target.view(-1, 1)
            output = model(seq, target, label_size)
            output = output.view(-1, label_size)
            rank_score = output.tolist()
            prediction_scores += rank_score
        if self.is_eval:
            df_test = pd.read_csv(os.path.join(self.root_path, 'sort_data/recall_result_for_eval.csv'))
        else:
            df_test = pd.read_csv(os.path.join(self.root_path, 'sort_data/recall_result.csv'))
        df_test['rank_scores'] = prediction_scores
        df_test['next_item_prediction'] = df_test.apply(lambda r:rank_item(r), axis=1)
        if self.is_eval:
            df_test.to_csv(os.path.join(self.root_path, 'sort_data/rank_result_for_eval.csv'), index=False)
        else:
            df_test.to_csv(os.path.join(self.root_path, 'sort_data/rank_result.csv'), index=False)


if __name__ == '__main__':
    p = PredNWPModel()
    p.process()
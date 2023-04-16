import os.path
import pickle
import torch.nn as nn
import torch
from word_dataset import WordDataSet
from transformer import NextWordPred
from transformer import Encoder
from preprocess_data import Vocabulary
from torch.autograd import Variable
from tqdm import tqdm

class EvalAndTestModel(object):
    def __init__(self, root_path='../../', is_eval=True):
        self.root_path = root_path
        self.is_eval = is_eval
        self.load_vocab()

    def load_vocab(self):
        if self.is_eval:
            vocab_path = os.path.join(self.root_path, 'rank_data/vocab_for_eval.pkl')
        else:
            vocab_path = os.path.join(self.root_path, 'rank_data/vocab.pkl')
        self.vocab = pickle.load(open(vocab_path, 'rb'))
        print("load vocab done")

    def process(self):
        if self.is_eval:
            data = WordDataSet(data_file='rank_data/test_for_eval.csv', root_path=self.root_path, type='eval')
        else:
            data = WordDataSet(data_file='rank_data/test.csv', root_path=self.root_path, type='test')

        eval_data = torch.utils.data.DataLoader(data, shuffle=False, batch_size=8)
        model = torch.load(os.path.join(self.root_path, 'rank_data/transformer_din_4.pth'))
        model.eval()
        prediction_scores = []
        for batch in tqdm(eval_data):
            x, y = batch
            x = x.view(-1, 21)
            output = model(x)
            output = output.view(-1, 100)
            rank_score = output.tolist()
            prediction_scores += rank_score
        data.add_prediction(prediction_scores)
        data.rerank()
        if self.is_eval:
            data.save(os.path.join(self.root_path, 'sort_data/rank_result_for_eval.csv'))


if __name__ == '__main__':
    e = EvalAndTestModel()
    e.process()
import os.path

import fasttext
import pandas as pd
from tqdm import tqdm
import pickle

class CandidatePooling(object):
    def __init__(self, model, size=10000, dst_file=None, num=10):
        self.model = model
        self.pool = []
        self.size = size
        self.dst_file = dst_file
        self.label = []
        self.seq = []
        self.num = num

    def get_pred(self):
        preds, scores = self.model.predict(self.pool, k=self.num)
        preds = [list(map(lambda k: k[9:], pred)) for pred in preds]
        scores = [score.tolist() for score in scores]
        return preds, scores

    def write_to_file(self):
        preds, scores = self.get_pred()
        pool = [p.split() for p in self.pool]
        for l, seq, p in zip(self.label, pool, preds):
            self.dst_file.write("{}\t{}\t{}\n".format(l, ",".join(seq), ",".join(p)))

    def add_instance(self, candidate, label=None):
        if label:
            self.label.append(label)
        self.pool.append(candidate)
        if len(self.pool) > self.size:
            if len(self.label) > 0:
                self.write_to_file()
                self.pool = []
                self.label = []
                return
            else:
                preds, scores = self.get_pred()
                self.pool = []
                self.label = []
                return preds, scores
        else:
            return None, None


class Fasttext(object):
    def __init__(self, root_path='../../', is_eval=True, **kwargs):
        self.root_path = root_path
        self.is_eval = is_eval
        self.has_train = kwargs['has_train']
        self.dim = kwargs['dim']
        self.train_file_path = os.path.join(self.root_path, 'input_data', kwargs['train_filename'])
        self.predict_for_rank_filename = os.path.join(self.root_path, 'input_data', kwargs['predict_for_rank_filename'])
        self.predict_dst_filename = os.path.join(self.root_path, 'input_data', kwargs['predict_dst_filename'])
        self.test_filename = os.path.join(self.root_path, 'input_data', kwargs['test_filename'])
        self.model_filename = os.path.join(self.root_path, 'recall_data', kwargs['model_filename'])
        self.num_for_train_sample = kwargs['num_for_train_sample']
        self.num_for_predict_sample = kwargs['num_for_eval_sample']

    def process(self):
        if self.has_train:
            model = fasttext.load_model(self.model_filename)
        else:
            model = fasttext.train_supervised(input=self.train_file_path, loss='hs', neg=100, minCountLabel=4, dim=self.dim, epoch=10)
            model.save_model(path=self.model_filename)
        print("gen train data for rank model based on fasttext model")
        predict_dst_file = open(self.predict_dst_filename, 'w')
        c = CandidatePooling(model, dst_file=predict_dst_file)
        for line in tqdm(open(self.predict_for_rank_filename, 'r')):
            line = line.strip().split('\t')
            label, seq = line
            seq = seq.split(",")
            c.add_instance(" ".join(seq), label)
        c.write_to_file()
        test_data = pd.read_csv(self.test_filename)
        prev_items = test_data['prev_items'].tolist()
        prev_items = list(map(eval, prev_items))
        prev_items = [" ".join(item) for item in prev_items]
        all_label = []
        fasttext_scores = []
        print("predict data based on fasttext model")
        for candidate in tqdm(prev_items, total=len(prev_items)):
            res, scores = model.predict(candidate, k=self.num_for_predict_sample)
            pred = list(map(lambda k:k[9:], res))
            fasttext_scores.append(scores.tolist())
            all_label.append(pred)
        test_data['next_item_prediction'] = all_label
        test_data['fasttext_scores'] = fasttext_scores
        if self.is_eval:
            test_data.to_csv(os.path.join(self.root_path, 'recall_data/fasttext_for_eval.csv'), index=False)
        else:
            test_data.to_csv(os.path.join(self.root_path, 'recall_data/fasttext.csv'), index=False)
        # if self.is_eval:
        #     test_data = pd.read_csv(os.path.join(self.root_path, "rank_data/test_for_eval.csv"))
        # else:
        #     test_data = pd.read_csv(os.path.join(self.root_path, "rank_data/test.csv"))



if __name__ == '__main__':
    f = Fasttext(hasTrain=False)
    f.process()
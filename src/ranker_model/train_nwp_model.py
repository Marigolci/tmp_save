import os.path
import sys
import torch.nn as nn
import torch
import numpy as np
from nwp_dataset import NWP_DataSet
from tqdm import tqdm
from simple_nwp_model import SimpleNWPModel


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

class Metric(object):
    def __init__(self):
        self.value = 0.0
        self.cnt = 0

    def update(self, v):
        self.value += v
        self.cnt += 1
    def get_avg(self):
        return self.value / self.cnt

    def zero(self):
        self.value = 0
        self.cnt = 0.0

class TrainNWPModel(object):
    def __init__(self, root_path='../../', is_eval=True, **kwargs):
        self.root_path = root_path
        self.is_eval = is_eval
        self.batch_size = kwargs['batch_size']
        self.word2idx = {}
        self.emb_size = kwargs['emb_size']
        word_dict_filename = os.path.join(self.root_path, 'input_data', kwargs['word_dict_filename'])
        for line in open(word_dict_filename, 'r'):
            line = line.strip()
            word, idx = line.split(":")
            self.word2idx[word] = int(idx)
        self.train_filename = os.path.join(self.root_path, 'input_data', kwargs['train_filename'])
        self.emb = np.load(os.path.join(self.root_path, 'recall_data', kwargs['emb_path']))
        self.model_filename = os.path.join(self.root_path, "rank_data", kwargs['model_filename'])

    def process(self):
        print("in train model process")
        print("pretrain emb shape is:", self.emb.shape)
        print("word dict size is {} emb size is:{}".format(len(self.word2idx)+10, self.emb_size))
        model = SimpleNWPModel(input_size=len(self.word2idx)+10, emb_size=self.emb_size, pretrain_emb=self.emb)
        print("model create done")
        # model = model.apply(initialize_weights)
        LEARNING_RATE = 0.05
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-8)
        criterion = torch.nn.CrossEntropyLoss()
        loss_metric = Metric()
        acc_metric = Metric()
        cur_cnt = 0
        train_file = open(self.train_filename, 'r')
        while True:
            loss_metric.zero()
            acc_metric.zero()
            train_dataset = NWP_DataSet(train_file, self.batch_size*2000)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size)
            for i, batch in tqdm(enumerate(train_dataloader)):
                seq, target, label = batch
                label_size = target.shape[1]
                target = target.view(-1, 1)
                label = label.view(-1)
                cur_batch_size = target.shape[0]
                # seq = seq.view(cur_batch_size, -1)
                optimizer.zero_grad()
                output = model(seq, target, label_size)
                output = output.view(-1, label_size)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                pred_idx = output.argmax(dim=1)
                acc = label.eq(pred_idx.view_as(label)).float().mean().item()
                # print(output, pred_idx, acc)
                # assert (0)
                loss_metric.update(loss.item())
                acc_metric.update(acc)
                if (i+1) % 100 == 0:
                    # print("output is:", output, "label is:", label)
                    print("current loss is:{}, current acc is:{}".format(loss_metric.get_avg(), acc_metric.get_avg()))
            print("current round {} is done".format(cur_cnt))
            cur_cnt += 1
            if train_dataset.is_last_file:
                break
        torch.save(model, self.model_filename)

if __name__ == '__main__':
    train = TrainNWPModel()
    train.process()
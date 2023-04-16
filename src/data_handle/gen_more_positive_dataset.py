import pandas as pd
import os
from tqdm import tqdm

class DataGenerationForMorePositive(object):
    def __init__(self, root_path='../../', is_eval=True):
        self.root_path = root_path
        self.is_eval = is_eval

    def gen_pos_data(self, sents, labels):
        pos_ins = {"prev_items": [], 'next_item': []}
        for prev_item, next_item in tqdm(zip(sents, labels)):
            if len(prev_item) > 1:
                for i in range(len(prev_item)-1):
                    pos_ins['prev_items'].append(prev_item[:i+1])
                    pos_ins['next_item'].append(prev_item[i+1])
        return pos_ins


    def process(self):
        df_train = pd.read_csv(os.path.join(self.root_path, 'input_data/train.csv'))
        if not self.is_eval:
            df_eval = pd.read_csv(os.path.join(self.root_path, 'input_data/eval.csv'))
            df_train = pd.concat([df_train, df_eval])
        print("gen more positive data")
        sents = df_train['prev_items'].tolist()
        sents = list(map(eval, sents))
        labels = df_train['next_item'].tolist()
        pos_ins = self.gen_pos_data(sents, labels)
        sents += pos_ins['prev_items']
        labels += pos_ins['next_item']
        if self.is_eval:
            output_filename_for_fasttext = os.path.join(self.root_path, 'input_data/train_for_fasttext_eval.bin')
            output_filename_for_rank = os.path.join(self.root_path, 'input_data/train_for_rank_more_pos_eval.bin')
        else:
            output_filename_for_fasttext = os.path.join(self.root_path, 'input_data/train_for_fasttext.bin')
            output_filename_for_rank = os.path.join(self.root_path, 'input_data/train_for_rank_more_pos.bin')
        print("gen data for fasttext")
        output_file = open(output_filename_for_fasttext, 'w')
        for s,l in tqdm(zip(sents, labels)):
            output_file.write("__label__{} {}\n".format(l, " ".join(s)))
        output_file.close()
        print("gen data for rank")
        output_file = open(output_filename_for_rank, 'w')
        for s,l in tqdm(zip(sents, labels)):
            output_file.write("{}\t{}\n".format(l, ",".join(s)))
        output_file.close()


if __name__ == "__main__":
    d = DataGenerationForMorePositive()
    d.process()
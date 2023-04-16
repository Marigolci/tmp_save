import os.path

import fasttext
import numpy as np
from tqdm import tqdm

class GetEmbedding(object):
    def __init__(self, is_eval=True, root_path='../../', **kwargs):
        self.is_eval = is_eval
        self.root_path = root_path
        self.word_dict_filename = kwargs['word_dict_filename']
        self.fasttext_model_name = kwargs['fasttext_model_name']
        self.emb_path = kwargs['emb_path']

    def process(self):
        vocab_filename = os.path.join(self.root_path, 'input_data', self.word_dict_filename)
        model = fasttext.load_model(os.path.join(self.root_path, "recall_data", self.fasttext_model_name))
        word_dict = {}
        for line in open(vocab_filename, 'r'):
            word, idx = line.strip().split(":")
            word_dict[word] = int(idx)
        emb_matrix = np.random.rand(len(word_dict)+10, 64)

        print("gen word embedding")
        for word in tqdm(word_dict):
            word_emb = model.get_word_vector(word)
            emb_matrix[word_dict[word],:] = word_emb
        np.save(os.path.join(self.root_path, 'recall_data', self.emb_path), emb_matrix)

if __name__ == '__main__':
    g = GetEmbedding()
    g.process()
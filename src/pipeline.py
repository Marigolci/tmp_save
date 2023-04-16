import os.path
import json
import argparse
import sys
sys.path.append("./ranker_model")
from ranker_model.train_nwp_model import TrainNWPModel
from ranker_model.pred_nwp_model import PredNWPModel
from gen_recall_data.dhot import Dhot
from gen_recall_data.ada_i2i import AdaI2I
from gen_recall_data.txt_i2i import TxtI2I
from sort.recall import RecallModule
from gen_recall_data.llr_i2i import LLRI2I
from data_handle.gen_dataset import DataGeneration
from data_handle.gen_rank_dataset import DataGenerationForRank
from data_handle.gen_more_positive_dataset import DataGenerationForMorePositive
from gen_recall_data.fasttext_model import Fasttext
from ranker_model.get_pretrain_embedding import GetEmbedding
from metric.recall_ratio import RecallRatio
from metric.mrr import Mrr
from sort.gen_submission import GenSubmission


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--is_eval', action="store_true")
parser.add_argument('-p', '--json_path', default='../config/eval_for_zwh.json')
args = parser.parse_args()

is_eval = args.is_eval
config = json.load(open(args.json_path,'r'))

def make_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


for step in config['steps']:
    print("run {} step".format(step))
    if not config[step]['enable']:
        print("skip {} step".format(step))
        continue
    if 'output_dir' in config[step]:
        make_dir(os.path.join('../', config[step]['output_dir']))
    if step == 'data_generation':
        current_module = DataGeneration(root_path='../')
    elif step == 'data_gen_more_positive':
        current_module = DataGenerationForMorePositive(is_eval=is_eval, root_path='../')
    elif step == 'ada_i2i':
        current_module = AdaI2I(is_eval=is_eval, root_path='../')
    elif step == 'ada_i2i_txt':
        current_module = TxtI2I(is_eval=is_eval, root_path='../')
    elif step == 'llr_i2i':
        current_module = LLRI2I(is_eval=is_eval, root_path='../')
    elif step == 'dhot':
        current_module = Dhot(is_eval=is_eval, root_path='../')
    elif step == 'fasttext_train':
        current_module = Fasttext(is_eval=is_eval, root_path='../', **config[step])
    elif step == 'recall' and config['recall']['enable']:
        current_module = RecallModule(is_eval=is_eval, root_path='../', **config[step])
    elif step == 'data_gen_for_rank':
        current_module = DataGenerationForRank(is_eval=is_eval, root_path='../', **config[step])
    elif step == 'get_pretrain_emb':
        current_module = GetEmbedding(is_eval=is_eval, root_path='../', **config[step])
    elif step == 'train_rank_model':
        current_module = TrainNWPModel(is_eval=is_eval, root_path='../', emb_size=config[step]['emb_size'])
    elif step == 'pred_rank_model':
        current_module = PredNWPModel(is_eval=is_eval, root_path='../', model_name=config[step]['model_name'])
    elif step == 'mmr_metric':
        current_module = Mrr(root_path='../', eval_file_name=config[step]['eval_filename'])
    elif step == 'gen_submission':
        current_module = GenSubmission(root_path='../', submit_file_name=config[step]['eval_filename'])
    elif step == 'recall_ratio_metric':
        current_module = RecallRatio(root_path='../', eval_file_name=config[step]['eval_filename'])
    else:
        continue
    current_module.process()


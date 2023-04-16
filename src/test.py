import sys
sys.path.append("./ranker_model")
from ranker_model.train_nwp_model import TrainNWPModel

t = TrainNWPModel(root_path="../")
t.process()
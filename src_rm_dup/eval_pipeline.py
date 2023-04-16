from gen_recall_data.dhot import Dhot
from gen_recall_data.ada_i2i import AdaI2I
from sort.recall import RecallModule
from metric.mrr import Mrr
from data_handle.gen_dataset import DataGeneration

# d = DataGeneration()
# d.process()
# a = AdaI2I(is_eval=True, root_path='../')
# a.process()
# b = Dhot(is_eval=True, root_path='../')
# b.process()
recall = RecallModule(is_eval=True, root_path='../')
recall.process()
mrr = Mrr(root_path='../')
mrr.process()
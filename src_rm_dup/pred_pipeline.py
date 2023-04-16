from gen_recall_data.dhot import Dhot
from gen_recall_data.ada_i2i import AdaI2I
from sort.recall import RecallModule
from gen_recall_data.dhot import Dhot
from sort.gen_submission import GenSubmission
from data_handle.gen_dataset import DataGeneration

# d = DataGeneration()
# d.process()
a = AdaI2I(is_eval=False, root_path='../')
a.process()
b = Dhot(is_eval=False, root_path='../')
b.process()
recall = RecallModule(is_eval=False, root_path='../')
recall.process()
g = GenSubmission(root_path='../')
g.process()


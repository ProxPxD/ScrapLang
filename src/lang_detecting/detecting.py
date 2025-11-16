from src.constants import Paths
from src.lang_detecting.preprocessing.data import DataPreprocessor
from src.lang_detecting.lang_predictor import LangPredictor
from src.resouce_managing.file import FileMgr

try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False

print(f'Torch is {("un", "")[has_torch]}available')
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))



class Detector:
    def __init__(self):
        ...

sp = DataPreprocessor()
preprocessed = sp.process(FileMgr(Paths.VALID_DATA_FILE).load())


# lang_script = sp.create_lang_script_correspondence()
# script_predictor = LangPredictor(lang_script)
# words = [
#     ['spać'],
#     ['мати'],
#     ['食べる'],
#     ['食'],
# ]
# for group in words:
#     pred = script_predictor.predict_lang(group)
#     print(pred)
#
#
# sp.create_script_set_model_groups()
from src.constants import Paths
from src.lang_detecting.preprocessing.data import DataPreprocessor
from src.lang_detecting.lang_predictor import LangPredictor
from src.resouce_managing.file import FileMgr


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
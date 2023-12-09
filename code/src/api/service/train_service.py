from os import path

import pandas as pd

from train.lpda import lpda, lpda_save_to_file
from data_utils import get_data_folder
from dataset import meta_heart, heart

class TrainService:
    @staticmethod
    def heart_train():
        a, b, _ = lpda(heart, meta_heart.y_column)
        lpda_save_to_file("heart_model.dat", a, b)
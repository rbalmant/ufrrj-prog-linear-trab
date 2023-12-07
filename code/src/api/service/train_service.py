from os import path

import pandas as pd

from train.lpda import lpda, lpda_save_to_file
from data_utils import get_data_folder
from dataset import heart

class TrainService:
    @staticmethod
    def heart_train():
        train = pd.read_csv(path.join(get_data_folder(), heart.folder, heart.file))
        a, b, _ = lpda(train, heart.y_column)
        lpda_save_to_file("model.dat", a, b)
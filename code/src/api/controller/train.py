from os import path
import logging

import pandas as pd
from fastapi import APIRouter

from train.lpda import lpda, lpda_save_to_file
from data_utils import get_data_folder

router = APIRouter()

@router.post("/train")
async def train_model():
    logging.debug(">> (api.train) train_model()")
    res = "OK"
    try:
        train = pd.read_csv(path.join(get_data_folder(), 'heart_disease', 'heart.csv'))
        a, b, _ = lpda(train, 'target')
        lpda_save_to_file("model.dat", a, b)
    except Exception as e:
        res = "ERROR"
        logging.error(e)
    logging.debug("<< (api.train) train_model()")
    return res


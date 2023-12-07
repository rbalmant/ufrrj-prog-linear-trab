from os import path
import pandas as pd


from fastapi import FastAPI

from api.controller.predict import router as predict_router
from api.controller.train import router as train_router

app = FastAPI()
app.include_router(predict_router)
app.include_router(train_router)

"""
from train.lpda import lpda, simple_predict
from data_utils import get_data_folder
def main():
    train = pd.read_csv(path.join(get_data_folder(), 'heart_disease', 'heart.csv'))

    a, b, X_test = lpda(train, 'target')

    print(a)
    print(b)

    for i in X_test:
        simple_predict(a, b, i) """



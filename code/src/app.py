from os import path
import pandas as pd

from fastapi import FastAPI

from api.controller.predict import router as predict_router
from api.controller.train import router as train_router

app = FastAPI()
app.include_router(predict_router)
app.include_router(train_router)

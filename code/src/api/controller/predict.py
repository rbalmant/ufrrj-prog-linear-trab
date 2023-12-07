from os import path
import logging

from fastapi import APIRouter

from train.lpda import lpda_load_from_file, simple_predict
from api.request.predict_request import PredictRequest

router = APIRouter()

@router.post("/predict")
async def predict(payload: PredictRequest):
    logging.debug(">> (api.predict) predict(payload: PredictRequest)")
    prediction = None
    if payload is not None:
        try:
            test = [vars(payload).values()]
            a, b = lpda_load_from_file("model.dat")
            prediction = simple_predict(a, b, test)
        except Exception as e:
            prediction = "ERROR"
            logging.error(e)
    else:
        prediction = "Please provide test data in the format of the PredictRequest"
    logging.debug("<< (api.train) predict(payload: PredictRequest)")
    return prediction
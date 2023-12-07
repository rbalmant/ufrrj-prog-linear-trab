import logging
from fastapi import APIRouter
from api.request.predict_request import HeartPredictRequest
from api.service.predict_service import PredictService

router = APIRouter()

@router.post("/heart/predict")
def predict(payload: HeartPredictRequest):
    logging.debug(">> (api.predict) predict(payload: PredictRequest)")
    prediction = None
    if payload is not None:
        try:
            prediction = PredictService.heart_predict(payload)
        except Exception as e:
            prediction = "ERROR"
            logging.error(e)
    else:
        prediction = "Please provide test data in the format of the PredictRequest"
    logging.debug("<< (api.train) predict(payload: PredictRequest)")
    return prediction
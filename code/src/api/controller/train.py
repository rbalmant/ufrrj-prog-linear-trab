import logging
from fastapi import APIRouter
from api.service.train_service import TrainService

router = APIRouter()

@router.post("/heart/train")
def train_model():
    logging.debug(">> (api.train) train_model()")
    res = "OK"
    try:
        TrainService.heart_train()
    except Exception as e:
        res = "ERROR"
        logging.error(e)
    logging.debug("<< (api.train) train_model()")
    return res



from api.request.predict_request import HeartPredictRequest
from train.lpda import lpda_load_from_file, simple_predict
from dataset import meta_heart

class PredictService:
    a, b = None, None

    @staticmethod
    def heart_predict(payload: HeartPredictRequest) -> int:
        test = [
            payload.age, 
            payload.sex, 
            payload.cp, 
            payload.trestbps, 
            payload.chol,
            payload.fbs,
            payload.restecg,
            payload.thalach,
            payload.exang,
            payload.oldpeak,
            payload.slope,
            payload.ca,
            payload.thal
        ]
        return simple_predict(meta_heart.solution[0], meta_heart.solution[1], test) 

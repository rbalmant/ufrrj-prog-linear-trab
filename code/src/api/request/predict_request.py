
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    age: int = Field(..., description="Age of the person")
    sex: int = Field(..., description="Gender of the person")
    cp: int = Field(..., description="Chest Pain type chest pain type")
    trestbps: int = Field(..., description="Resting blood pressure (in mm Hg)")
    chol: int = Field(..., description="Cholestoral in mg/dl fetched via BMI sensor")
    fbs: int = Field(..., description="(fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)")
    restecg: int = Field(..., description="resting electrocardiographic results")
    thalach: int = Field(..., description="maximum heart rate achieved")
    exang: int = Field(..., description="exercise induced angina (1 = yes; 0 = no)")
    oldpeak: float = Field(..., description="Previous peak")
    slope: int = Field(..., description="Slope")
    ca: int = Field(..., description="number of major vessels (0-3)")
    thal: int = Field(..., description="Thal rate")

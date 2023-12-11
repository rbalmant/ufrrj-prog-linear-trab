from model import model
from dataset import heart, meta_heart

model_result = model(heart, meta_heart.y_column)
print(model_result)
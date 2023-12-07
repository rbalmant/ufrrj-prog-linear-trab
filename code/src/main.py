from io import StringIO
import pandas as pd
import numpy as np

from train.lpda import lpda
from utils import unzip_files_into_memory

files = unzip_files_into_memory('/data/gladiator_data.zip')
dataset = files[0] # only 1 file in ZIP file

df = pd.read_csv(StringIO(dataset))


model, X_test = lpda(df, 'Survived')

# Validate model

# Predict, plot, etc



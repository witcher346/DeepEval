from keras.models import model_from_json, load_model
import numpy as np
from sklearn.externals.joblib import load
import tensorflow as tf
import keras
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

loaded_model = load_model('_model/model.h5')
print("Loaded model from disk")

while True:
    print('''
This is an interface to use neural network for apartments prices prediction or DeepEval.
Type in one of the following commands:
1 - Input parameters for neural network
2 - Exit
    ''')
    user_answer = int(input())
    if user_answer == 1:

        koatauu = int(input('Input koatauu: '))
        kitchen_area = int(input('Input kitchen_area: '))
        qt_room = int(input('Input number of rooms: '))
        floor = int(input('Input apartments floor: '))
        qt_floors = int(input('Input number of floors in a building: '))
        total_area = int(input('Input total area: '))
        living_area = int(input('Input living area: '))
        year_built = int(input('Input the year when building was built: '))

        input_scaler = load('_model/input_scaler.bin')
        output_scaler = load('_model/output_scaler.bin')

        data = np.asarray([koatauu, kitchen_area, qt_room, floor, qt_floors, total_area, living_area, year_built], dtype=np.float64)
        data = input_scaler.transform(data.reshape((1,-1)))

        prediction = loaded_model.predict(data)
        prediction = output_scaler.inverse_transform(prediction)
        print('Estimated price by a model: {}'.format(prediction[0][0]))

    else:
        break

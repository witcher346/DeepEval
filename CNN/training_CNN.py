from keras.layers import Conv2D, MaxPool2D, Dense, BatchNormalization, Flatten, Dropout
from keras.optimizers import Adam
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle

train_data_df = pd.read_excel('train_data_combined_cnn.xlsx')
'''
y_label = np.asarray(train_data_df['price'])
y_label = y_label.reshape((-1,1))
train_scaler_Y = StandardScaler().fit(y_label)
'''
scaler = max(train_data_df['price'])
train_data_df['price'] = train_data_df['price']/scaler

test_data_df = pd.read_excel('test_data_combined_cnn.xlsx')
'''
test_y_label = np.asarray(test_data_df['price'])
test_y_label = test_y_label.reshape((-1,1))
test_scaler_Y = StandardScaler().fit(test_y_label)
'''
test_data_df['price'] = test_data_df['price']/scaler

datagen = ImageDataGenerator(rescale=1/255)
train_data = datagen.flow_from_dataframe(dataframe=train_data_df, x_col='filepath', y_col='price', class_mode='raw', directory=r'C:\Users\Kojimba\PycharmProjects\DeepEval\CNN', batch_size=5)
test_data = datagen.flow_from_dataframe(dataframe=train_data_df, x_col='filepath', y_col='price', class_mode='raw', directory=r'C:\Users\Kojimba\PycharmProjects\DeepEval\CNN', batch_size=1)

model = Sequential([
    Conv2D(128, kernel_size=8, strides=6, padding='same', activation='relu', input_shape=(256, 256, 3), data_format='channels_last'),
    Dropout(0.3),
    MaxPool2D(strides=2),
    Conv2D(256, kernel_size=8, strides=6, padding='same', activation='relu'),
    Dropout(0.2),
    MaxPool2D(pool_size=4),
    Flatten(),
    Dense(1, activation='linear', kernel_initializer='random_normal', bias_initializer='zeros')
])

model.compile(Adam(lr=0.001, beta_1=0.97, beta_2=0.998), loss='mse')
model.summary()

model.fit_generator(train_data, steps_per_epoch=25, epochs=25)

predict = model.predict_generator(test_data, steps=24)
#mse = mean_squared_error(test_y_label, predict)

test_data_df['price'] *= scaler
predict *= scaler
print(predict)
print(test_data_df['price'])
print('Qt prediction {}'.format(predict.shape[0]))
print('Mean in prediction: {}, mean in labels: {}'.format(predict.mean(), np.mean(test_data_df['price'])))

model.save(r'C:\Users\Kojimba\PycharmProjects\DeepEval\_trained_model\CNN.h5')
print('Model saved to disk')

with open(r'C:\Users\Kojimba\PycharmProjects\DeepEval\_trained_model\CNN_scaler.bin', 'wb') as file:
    pickle.dump(scaler, file)
print('Saved scaler to disk')
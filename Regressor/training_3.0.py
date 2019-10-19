import pandas as pd
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.externals.joblib import dump

df = pd.read_excel('data2.xlsx', sheet_name='Лист1')
df = df.sample(frac=1, random_state=220).reset_index(drop=True)

train_data = df[['koatuu_code', 'kitchen_area', 'qt_room', 'floor', 'qt_floor', 'total_area', 'living_area', 'id_wall_material']][0:43620].values
y_label = df['price_usd'][0:43620].values

pred_test = df[['koatuu_code', 'kitchen_area', 'qt_room', 'floor', 'qt_floor', 'total_area', 'living_area', 'id_wall_material']][43620:].values
pred_y_label = df['price_usd'][43620:].values

scale_train_X = StandardScaler()
scale_train_Y = StandardScaler()

train_data = scale_train_X.fit_transform(train_data)
y_label = scale_train_Y.fit_transform(y_label.reshape((-1,1)))

pred_test = scale_train_X.fit_transform(pred_test)
pred_y_label = scale_train_Y.transform(pred_y_label.reshape((-1,1)))

dump(scale_train_X, 'input_scaler.bin', compress=True)
dump(scale_train_Y, 'output_scaler.bin', compress=True)


def build_regression():

    model = Sequential([
        Dense(128, activation='relu', input_shape=(8,), kernel_initializer='random_normal', bias_initializer='Zeros'),
        #BatchNormalization(),
        Dropout(0.1),
        Dense(32, activation='relu', kernel_initializer='random_normal', bias_initializer='Zeros'),
        #BatchNormalization(),
        Dense(64, activation='relu', kernel_initializer='random_normal', bias_initializer='Zeros'),
        Dropout(0.3),
        Dense(1, activation='linear', kernel_initializer='random_normal', bias_initializer='Zeros'),
        ])

    opt = Adam(lr=0.00005, beta_1=0.97, beta_2=0.999) #RMSprop(lr=0.0001)
    model.compile(optimizer=opt, loss='mse')
    return model

estimator = build_regression()#KerasRegressor(build_regression, batch_size=16, epochs=100, validation_split=0.3)

estimator.fit(x=train_data, y=y_label, batch_size=32, epochs=150, validation_split=0.25, shuffle=False, callbacks=[EarlyStopping(monitor='loss', patience=5)])

predict = estimator.predict(pred_test)
mse = mean_squared_error(pred_y_label, predict)

predict = scale_train_Y.inverse_transform(predict)
pred_y_label = scale_train_Y.inverse_transform(pred_y_label)

diff = predict - pred_y_label
percDiff = (diff / pred_y_label) * 100
absPercDiff = np.abs(percDiff)
mean_perc = np.mean(absPercDiff)
std_perc = np.std(absPercDiff)

print('Mean error percentage: {} \nStandard error percentage: {}\nMSE: {}'.format(mean_perc, std_perc, mse))
print('Mean in prediction: {}, mean in labels: {}'.format(predict.mean(), pred_y_label.mean()))

print(predict.T)
print(pred_y_label.T)

plt.plot(absPercDiff, 'ob')
plt.ylabel("Labeled vals")
plt.xlabel("Predicted vals")
plt.show()


model_json = estimator.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
estimator.save_weights("model.h5")
print("Saved model to disk")

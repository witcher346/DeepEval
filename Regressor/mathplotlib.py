import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import random
from sklearn.preprocessing import StandardScaler
from scipy import signal


matplotlib.use('TkAgg')

df = pd.read_excel('data2.xlsx', sheet_name='Лист1')
df = df.sample(frac=1, random_state=220).reset_index(drop=True)

df = df['price_usd']
df = df.loc[0:].values
plt.plot(df, 'bo')
plt.ylabel('')
plt.xlabel('Prices')
plt.show()

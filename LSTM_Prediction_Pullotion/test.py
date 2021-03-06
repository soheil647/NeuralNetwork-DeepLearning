from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv, date_range
from pandas import DataFrame
from pandas import concat
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM, GRU, SimpleRNN, Average
import time
import numpy as np
import pandas as pd


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# load dataset
dataset = read_csv('pollution.csv', header=0, index_col=0)
dataset_no_miss = read_csv('pollution.csv', header=0, index_col=0)
# MAke 20 percent of each column miss
for key, value in dataset.iteritems():
    dataset.loc[dataset[key].sample(frac=.2).index, key] = np.nan

print(dataset['dew'].isna().sum())
dataset[pd.isnull(dataset['wnd_dir'])]= 'NaN'
print(dataset)

# dataset = dataset[:15000]
# print(dataset)

# find one hour of day in week
# random = np.random.randint(0, 23)
# random_day = np.random.randint(1, 31)
# dataset = dataset.iloc[random::24, :]
# dataset = dataset.iloc[random_day::31, :]


def find_corolation(df):
    import pandas as pd
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    df = DataFrame(df)
    df['wnd_dir'] = df['wnd_dir'].astype('category').cat.codes
    print(df.corr())


# find_corolation(dataset)

# Feature Selection : select 2
# dataset = dataset[['pollution', 'dew', 'wnd_dir']]

values = dataset.values
# integer encode direction
encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# specify the number of lag hours
n_hours = 11
n_features = 8

# Fill Missing Data
df = DataFrame(scaled)
print(df.mean())
df = df.fillna(df.mean())
print(df)

# find Mean Square Error
values = dataset_no_miss.values
# integer encode direction
encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# Mean Square Error
df_new = DataFrame(scaled)

print("pollution MSE: ", mean_squared_error(df.iloc[:, 0], df_new.iloc[:, 0]))
print("dew MSE: ", mean_squared_error(df.iloc[:, 1], df_new.iloc[:, 1]))
print("temp MSE: ", mean_squared_error(df.iloc[:, 2], df_new.iloc[:, 2]))
print("press MSE: ", mean_squared_error(df.iloc[:, 3], df_new.iloc[:, 3]))
print("wnd_dir: ", mean_squared_error(df.iloc[:, 4], df_new.iloc[:, 4]))
print("wnd_speed MSE: ", mean_squared_error(df.iloc[:, 5], df_new.iloc[:, 5]))
print("snow MSE: ", mean_squared_error(df.iloc[:, 6], df_new.iloc[:, 6]))
print("rain MSE: ", mean_squared_error(df.iloc[:, 7], df_new.iloc[:, 7]))

# frame as supervised learning
reframed = series_to_supervised(df.values, n_hours, 1)
print(reframed)
# split into train and test sets
values = reframed.values
split = int(len(dataset.index) * 0.8)
train = values[:split, :]
test = values[split:, :]

# split into input and outputs
n_obs = n_hours * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]
print(train_X.shape, len(train_X), train_y.shape)

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(GRU(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='rmsprop', metrics=['mse'])
# fit network
start = time.time()
history = model.fit(train_X, train_y, epochs=20, batch_size=64, validation_split=0.2, verbose=2,
                    shuffle=False)
print("Elapsed Time for fitiing: ", time.time()-start)

##################################################
# fusion models
# n_members = 3
# models = list()
# for i in range(n_members):
#     model1 = Sequential()
#     model1.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]), recurrent_dropout=0.3))
#     model1.add(Dense(1))
#     model1.compile(loss='mae', optimizer='rmsprop', metrics=['mse'])
#     history1 = model1.fit(train_X, train_y, epochs=100, batch_size=64, validation_split=0.2, verbose=2, shuffle=False)
#     models.append(model1)
# # make a prediction
# yhat = [model.predict(train_X) for model in models]
# yhat = np.array(yhat)
# yhat = np.mean(yhat)
# print(yhat)
# train_X = train_X.reshape((train_X.shape[0], n_hours * n_features))
# # invert scaling for forecast
# inv_yhat = concatenate((yhat, train_X[:, -7:]), axis=1)
# inv_yhat = scaler.inverse_transform(inv_yhat)
# inv_yhat = inv_yhat[:, 0]
# # invert scaling for actual
# train_y = train_y.reshape((len(train_y), 1))
# inv_y = concatenate((train_y, train_X[:, -7:]), axis=1)
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:, 0]
#####################################################################


# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='validation')
pyplot.legend()
pyplot.show()

# plot history
# pyplot.plot(history.history['accuracy'], label='train')
# pyplot.plot(history.history['val_accuracy'], label='validation')
# pyplot.legend()
# pyplot.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_hours * n_features))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, -7:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -7:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]

# # make prediction low data
# yhat = model.predict(train_X)
# train_X = train_X.reshape((train_X.shape[0], n_hours * n_features))
# # invert scaling for forecast
# inv_yhat = concatenate((yhat, train_X[:, -7:]), axis=1)
# inv_yhat = scaler.inverse_transform(inv_yhat)
# inv_yhat = inv_yhat[:, 0]
# # invert scaling for actual
# train_y = train_y.reshape((len(train_y), 1))
# inv_y = concatenate((train_y, train_X[:, -7:]), axis=1)
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:, 0]

# plot history
pyplot.plot(inv_yhat, 'bo', label='predict')
pyplot.plot(inv_y, 'yo', label='real')
pyplot.legend()
pyplot.show()

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

time = '2011-05-18 00:00:' + str(1)
dates_12 = date_range('2011-05-17 12:00:00', periods=3000)
dates_24 = date_range(time, periods=1400)
timestampStr_12 = dates_12.strftime("%Y-%m-%d %H:%M:%S")
timestampStr_24 = dates_24.strftime("%Y-%m-%d %H:%M:%S")

from prettytable import PrettyTable

t = PrettyTable(['Date Hour 12', 'Predicted 12', 'Real 12', 'Date Hour 0', 'Predicted 0', 'Real 0'])
j = 0
for i in range(len(timestampStr_24)):
    if j > 3000:
        break
    t.add_row([timestampStr_24[i], inv_yhat[j + 24], inv_y[j + 24]])
    # t.add_row([timestampStr_24[i], inv_yhat[j + 12], inv_y[j + 12]])
    # print(timestampStr_12[i], ": ", 'Predicted: ', inv_yhat[j], ' Real: ', inv_y[j], end='  ')
    # print(timestampStr_24[i], ": ", 'Predicted: ', inv_yhat[j + 12], ' Real: ', inv_y[j + 12])
    j += 24
table_txt = t.get_string()
# with open('GRU_RMSprop_MAE_Weekly_Series.txt', 'w') as file:
#     file.write(table_txt)
# print(t)

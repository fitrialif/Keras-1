import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#--------------------------------------------------------------------------------------#

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

#--------------------------------------------------------------------------------------#


dataframe   = pandas.read_csv('./data/new_houses_london.csv')
dataframe   = dataframe["New Dwellings London"]
dataset     = dataframe.values
dataset     = dataset.astype('float32')
dataset     = dataset[:,np.newaxis]


scaler      = MinMaxScaler(feature_range=(0, 1))
dataset     = scaler.fit_transform(dataset)
train_size  = int(len(dataset) * 0.67)
test_size   = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(train.shape)
print(test.shape)


look_back      = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY   = create_dataset(test, look_back)
print(trainX.shape)
print(testX.shape)

trainX         = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX          = np.reshape(testX, (testX.shape[0]  , 1, testX.shape[1] ))

#--------------------------------------------------------------------------------------#


model = Sequential()
model.add(LSTM(4, input_dim=look_back, return_sequences=True))
model.add(LSTM(4, input_dim=look_back))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=100, batch_size=1, verbose=2)


print(trainX.shape)
print(testX.shape)


trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

#-------------------------------------------------------------------------------------#

np.savetxt('dataset_orig.csv',scaler.inverse_transform(dataset), delimiter=',')
np.savetxt('trainpred.csv'   ,trainPredict                     , delimiter=',')
np.savetxt('testpred.csv'    ,testPredict                      , delimiter=',')
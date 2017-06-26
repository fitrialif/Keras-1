#import os    
#os.environ['THEANO_FLAGS'] = "device=cpu"

import pandas
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, merge,Input
from keras.optimizers import SGD
from keras.preprocessing import text
import numpy as np

######################################################################################
#Processing data

g = pandas.read_csv("C:/Udemy/Pro Data Science in Python/\HousePricesKingCounty/kc_house_data.csv",
                    encoding = "ISO-8859-1")
g["price"] = g["price"]/1000


X = g[["sqft_above","sqft_basement","sqft_lot","sqft_living","floors",
       "bedrooms","yr_built","lat","long","bathrooms"]].values
Y = g["price"].values
zipcodes        = pandas.get_dummies(g["zipcode"]).values
condition       = pandas.get_dummies(g["condition"]).values
grade           = pandas.get_dummies(g["grade"]).values

X = np.concatenate((X,zipcodes),axis=1)
X = np.concatenate((X,condition),axis=1)
X = np.concatenate((X,grade),axis=1)

#######################################################################################
#Building deep network

 
a = Input(shape=(10,))
c = Dense(32, activation='tanh')(a)


b = Input(shape=(87,))
d = Dense(32, activation='tanh')(b)


e = merge([a, b], mode='concat')
finak = Dense(1)(e)

model = Model(input=[a, b,], output=finak)

#compiling and running
model.compile(optimizer='rmsprop', loss='mae')
model.fit([X[:,0:10], X[:,10:97]], Y ,nb_epoch=5000, batch_size=32, verbose=2)

g["predicted"] = model.predict([X[:,0:10], X[:,10:97]])

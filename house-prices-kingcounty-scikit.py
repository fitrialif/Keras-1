#import os    
#os.environ['THEANO_FLAGS'] = "device=cpu"

import pandas
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.preprocessing import text
import numpy as np
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn import cross_validation

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

def neural_model1():
    model = Sequential()
    model.add(Dense(50, input_dim=97, init='normal', activation='relu'))
    model.add(Dense(5, init='normal',activation="relu"))
    model.add(Dropout(0.05))
    model.add(Dense(5, init='normal',activation="relu"))
    model.add(Dense(1, init='normal'))
    sgd = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mae',optimizer=sgd,metrics=["mae"])
    return model


def neural_model2():
    model = Sequential()
    model.add(Dense(50, input_dim=97, init='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(14, init='normal',activation="sigmoid"))
    model.add(Dropout(0.3))
    model.add(Dense(5, init='normal',activation="relu"))
    model.add(Dense(1, init='normal'))
    sgd = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mae',optimizer=sgd,metrics=["mae"])
    return model
  
 
#model.fit(X,Y,nb_epoch=300,verbose=2)

#######################################################################################

model = KerasRegressor(build_fn=neural_model1, nb_epoch=10, batch_size=50, verbose=0)
results = cross_validation.cross_val_score(model, X, Y, cv=5, scoring = "mean_absolute_error")
print("Model1:" + str(results.mean()))


model = KerasRegressor(build_fn=neural_model2, nb_epoch=10, batch_size=50, verbose=0)
results = cross_validation.cross_val_score(model, X, Y, cv=5, scoring = "mean_absolute_error")
print("Model2:" + str(results.mean()))

#model.fit(X,Y)


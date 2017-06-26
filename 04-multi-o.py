import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input
from keras.optimizers import SGD
from keras.preprocessing import text
import numpy as np
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler

############################################################################

data = pd.read_csv("./data/Video_Games_Sales_as_at_22_Dec_2016.csv")
print(data.values.shape)

data = data.loc[data["NA_Sales"]>1]
data = data.loc[data["EU_Sales"]>1]

data = data.dropna(axis=0)
print(data.values.shape)
names = data[["Name","Year_of_Release"]]

NA = data["NA_Sales"].values
EU = data["EU_Sales"].values
JP = data["JP_Sales"].values
X  = data[["Critic_Score","Critic_Count","User_Count","Year_of_Release"]].values

platform  = pd.get_dummies(data["Platform"]).values
genre     = pd.get_dummies(data["Genre"]).values
publisher = pd.get_dummies(data["Publisher"]).values
rating    = pd.get_dummies(data["Rating"]).values


X = np.concatenate((X,platform,genre,publisher,rating),axis=1)
############################################################################

a = Input(shape=(60,))
c = Dense(32, activation='sigmoid')(a)
d = Dense(32, activation='sigmoid')(c)
fina1 = Dense(1,activation="linear")(d)
fina2 = Dense(1,activation="linear")(d)
fina3 = Dense(1,activation="linear")(d)
model = Model(input=a, output=[fina1,fina2,fina3])


model.compile(optimizer='adam', loss='mse')
model.fit(X, [NA,EU,JP] ,nb_epoch=3000, batch_size=100, verbose=2)
p_NA,p_EU,p_JP = model.predict(X)


predictions = pd.DataFrame(np.concatenate((names,p_NA,NA[:, np.newaxis],p_EU,EU[:, np.newaxis],p_JP,JP[:, np.newaxis]),axis=1))
predictions.columns = (["Name","Year","p_NA","NA","p_EU","EU","p_JP","JP"])


print(predictions)

############################################################################


import cv2

import Matrix_CV_ML3D as DImage
from keras import backend as K
import numpy as np
K.set_image_dim_ordering('th')


x = DImage.Matrix_CV_ML3D("./data/RiverStreet_train",100,133)
x.build_ML_matrix()


###################################################################################################


import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.datasets import cifar10


y = np_utils.to_categorical(x.labels)
x = x.global_matrix
x = x.astype('float32')/255

model = Sequential()
 
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(3,133,100)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
   
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
 
# 8. Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 
# 9. Fit model on training data
model.fit(x, y, batch_size=32, nb_epoch=20, verbose=2)

print(model.summary())

score = model.evaluate(x, y, verbose=0)

##################################################################################################

test = DImage.Matrix_CV_ML3D("./data/RiverStreet_test",100,133)
test.build_ML_matrix()

labels = test.labels
y = np_utils.to_categorical(test.labels)
test = test.global_matrix
test = test.astype('float32')/255

score = model.evaluate(test, y, verbose=0)
pred  = model.predict(test)
predicted = np.argmax(pred,axis=1)

labels    = pd.Series(labels).to_frame("Labels")
predicted = pd.Series(predicted).to_frame("Predicted")
p = pd.concat((labels,predicted),axis=1)
print(p)

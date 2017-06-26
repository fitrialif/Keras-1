#import os    
#os.environ['THEANO_FLAGS'] = "device=cpu"

import pandas
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,LocallyConnected1D,Reshape,Conv1D,Flatten
from keras.optimizers import SGD
import numpy as np
import Matrix_CV_ML as ML
from keras import backend as K
import numpy as np
K.set_image_dim_ordering('th')

######################################################################################

g = ML.Matrix_CV_ML("C:/Udemy/Pro Data Science in Python/sq_vs_tri/tri",10,13)
g.build_ML_matrix()


data_dim = 10*13
nb_classes = 2
g.global_matrix = g.global_matrix.reshape(24,130,1)

 
model = Sequential()

model.add(Conv1D(nb_filter = 10, filter_length=2, input_shape = (130,1)))
model.add(Flatten())

model.add(Dropout(0.3))
model.add(Dense(20, activation='relu'))

model.add(Reshape((20,1)))
model.add(LocallyConnected1D(nb_filter = 10, filter_length=2, input_shape = (20,1)))
model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='binary_crossentropy',
              optimizer=sgd,  
              metrics=["accuracy"])  
 
model.fit(g.global_matrix, g.labels,nb_epoch=100, batch_size=1, verbose=2)

print(model.evaluate(g.global_matrix, g.labels))


#######################################################################################

#j = ML.Matrix_CV_ML("C:/Udemy/Pro Data Science in Python/sq_vs_tri/test",100,133)
#j.build_ML_matrix()

#preds = (model.predict(j.global_matrix) > 0.5).flatten()

#data = np.transpose((np.array([j.labels,preds])))

#d = pandas.DataFrame(data,columns=["Observed","Predicted"])


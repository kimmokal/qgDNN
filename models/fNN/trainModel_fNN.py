# Restrict to one GPU
import imp
try:
	imp.find_module('setGPU')
	import setGPU
except ImportError:
	found = False

# Import modules
import tensorflow as tf
sess = tf.Session()
import matplotlib.pyplot as plt
import keras.backend as K
K.set_session(sess)
import keras.callbacks
import numpy as np
import pandas as pd
import root_pandas
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Define the path to the working directory
workPath = "/work/kimmokal/qgDNN"

### Choose the eta, pT bin for training
databin = "eta1.3_pt30to100"
# databin = "eta1.3_pt100to300"
# databin = "eta1.3_pt300to1000"
# databin = "eta1.3_pt1000"
# databin = "eta2.5_pt30to100"
# databin = "eta2.5_pt100to300"
# databin = "eta2.5_pt300to1000"

print 'Bin: ', databin

# Read in the data sets
trainSetPath = workPath+"/data/trainSets/fNN/preprocessed_FNN_trainSet_"+databin+".root"
testSetPath = workPath+"/data/testSets/fNN/preprocessed_FNN_testSet_"+databin+".root"

df_train = root_pandas.read_root(trainSetPath, 'tree')
df_test = root_pandas.read_root(testSetPath, 'tree')

# Extract the labels
train_y = df_train.isPhysUDS.copy()
test_y = df_test.isPhysUDS.copy()

train_x = df_train
train_x.drop(['isPhysUDS','jetQGl'], axis=1, inplace=True)
test_x = df_test
test_x.drop(['isPhysUDS','jetQGl'], axis=1, inplace=True)

# Convert the dataframes to matrices
train_x = train_x.as_matrix()
train_y = train_y.as_matrix()
test_x = test_x.as_matrix()
test_y = test_y.as_matrix()

# Build the neural network
from keras.models import Sequential,Model
from keras.layers import Input,Dense,Activation,Dropout
from keras.constraints import maxnorm
from keras import optimizers
from sklearn.utils import class_weight

model = Sequential()
model.add(Dense(100, kernel_initializer='normal', activation='relu', input_dim=train_x.shape[1]))
model.add(Dropout(0.2))
model.add(Dense(100, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

optimizer_ = 'Nadam'
loss_ = 'binary_crossentropy'
model.compile(optimizer=optimizer_, loss=loss_, metrics=['accuracy'])

earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=4)
classWeight = class_weight.compute_class_weight('balanced', np.unique(train_y), train_y[:])
batchSize = 512
numberOfEpochs = 30

model.fit(train_x, train_y,
        epochs=numberOfEpochs,
        batch_size = batchSize,
        class_weight=classWeight,
        callbacks=[earlystop],
        validation_split=0.2,
        shuffle=True)

# Save the trained model
savePath = workPath+'/models/trainedModels/'
model.save(savePath+'FNN_model_'+databin+'.h5')

# Print final ROC AUC score
print ' - roc auc: ',round(roc_auc_score(test_y,model.predict(test_x)),3)

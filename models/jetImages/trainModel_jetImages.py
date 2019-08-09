# Restrict to one GPU
import imp
try:
	imp.find_module('setGPU')
	import setGPU
except ImportError:
	found = False

# Import modules
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
import keras.backend as K
import keras.callbacks
import numpy as np
import pandas as pd
import root_pandas
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
sess = tf.Session()
K.set_session(sess)

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
trainSetQuarkPath = [workPath+"/data/trainSets/jetImages/jetImagesQuark_trainSet_"+databin+".h5"]
trainSetGluonPath = [workPath+"/data/trainSets/jetImages/jetImagesGluon_trainSet_"+databin+".h5"]

testSetQuarkPath = [workPath+"/data/testSets/jetImages/jetImagesQuark_testSet_"+databin+".h5"]
testSetGluonPath = [workPath+"/data/testSets/jetImages/jetImagesGluon_testSet_"+databin+".h5"]

# Create the channels
trainQuarkNtPt = []
trainQuarkChPt = []
trainQuarkChMult = []
trainGluonNtPt = []
trainGluonChPt = []
trainGluonChMult = []

testQuarkNtPt = []
testQuarkChPt = []
testQuarkChMult = []
testGluonNtPt = []
testGluonChPt = []
testGluonChMult = []

# Read the jet images and separate the channels (there is a for loop because there used to be many files to be iterated over)
for idx, gluonFile in enumerate(trainSetGluonPath):
	h5g = h5py.File(gluonFile, 'r')
	if idx == 0:
		trainGluonNtPt = h5g['NtPt'][()]
		trainGluonChPt = h5g['ChPt'][()]
		trainGluonChMult = h5g['ChMult'][()]
	else:
		trainGluonNtPt = np.append(trainGluonNtPt, h5g['NtPt'][()], axis=0)
		trainGluonChPt = np.append(trainGluonChPt, h5g['ChPt'][()], axis=0)
		trainGluonChMult = np.append(trainGluonChMult, h5g['ChMult'][()], axis=0)

for idx, quarkFile in enumerate(trainSetQuarkPath):
	h5g = h5py.File(quarkFile, 'r')
	if idx == 0:
		trainQuarkNtPt = h5g['NtPt'][()]
		trainQuarkChPt = h5g['ChPt'][()]
		trainQuarkChMult = h5g['ChMult'][()]
	else:
		trainQuarkNtPt = np.append(trainQuarkNtPt, h5g['NtPt'][()], axis=0)
		trainQuarkChPt = np.append(trainQuarkChPt, h5g['ChPt'][()], axis=0)
		trainQuarkChMult = np.append(trainQuarkChMult, h5g['ChMult'][()], axis=0)

for idx, gluonFile in enumerate(testSetGluonPath):
	h5g = h5py.File(gluonFile, 'r')
	if idx == 0:
		testGluonNtPt = h5g['NtPt'][()]
		testGluonChPt = h5g['ChPt'][()]
		testGluonChMult = h5g['ChMult'][()]
	else:
		testGluonNtPt = np.append(testGluonNtPt, h5g['NtPt'][()], axis=0)
		testGluonChPt = np.append(testGluonChPt, h5g['ChPt'][()], axis=0)
		testGluonChMult = np.append(testGluonChMult, h5g['ChMult'][()], axis=0)

for idx, quarkFile in enumerate(testSetQuarkPath):
	h5g = h5py.File(quarkFile, 'r')
	if idx == 0:
		testQuarkNtPt = h5g['NtPt'][()]
		testQuarkChPt = h5g['ChPt'][()]
		testQuarkChMult = h5g['ChMult'][()]
	else:
		testQuarkNtPt = np.append(testQuarkNtPt, h5g['NtPt'][()], axis=0)
		testQuarkChPt = np.append(testQuarkChPt, h5g['ChPt'][()], axis=0)
		testQuarkChMult = np.append(testQuarkChMult, h5g['ChMult'][()], axis=0)

# Create the labels: 1 for quark; 0 for gluon
trainQuark_y = np.empty(trainQuarkNtPt.shape[0], dtype=int); trainQuark_y.fill(1)
testQuark_y = np.empty(testQuarkNtPt.shape[0], dtype=int); testQuark_y.fill(1)

trainGluon_y = np.empty(trainGluonNtPt.shape[0], dtype=int); trainGluon_y.fill(0)
testGluon_y = np.empty(testGluonNtPt.shape[0], dtype=int); testGluon_y.fill(0)

# Combine the quark and gluon jets per channel
train_NtPt = np.concatenate((trainQuarkNtPt, trainGluonNtPt), axis=0)
train_ChPt = np.concatenate((trainQuarkChPt, trainGluonChPt), axis=0)
train_ChMult = np.concatenate((trainQuarkChMult, trainGluonChMult), axis=0)
test_NtPt = np.concatenate((testQuarkNtPt, testGluonNtPt), axis=0)
test_ChPt = np.concatenate((testQuarkChPt, testGluonChPt), axis=0)
test_ChMult = np.concatenate((testQuarkChMult, testGluonChMult), axis=0)

train_y = np.concatenate((trainQuark_y, trainGluon_y))
test_y = np.concatenate((testQuark_y, testGluon_y))

# Shuffle the training data and labels accordingly
randPerm = np.random.RandomState(seed=42).permutation(len(train_y))
np.take(train_NtPt, indices=randPerm, axis=0, out=train_NtPt)
np.take(train_ChPt, indices=randPerm, axis=0, out=train_ChPt)
np.take(train_ChMult, indices=randPerm, axis=0, out=train_ChMult)
train_y = train_y[randPerm]

# Merge the channels
train_x = np.stack((train_NtPt, train_ChPt, train_ChMult), axis=-1)
test_x = np.stack((test_NtPt, test_ChPt, test_ChMult), axis=-1)

# Build the neural network
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Activation, Dropout, Flatten
from keras import optimizers
from sklearn.utils import class_weight

model = Sequential()
model.add(Convolution2D(filters=64, kernel_size=(8,8), kernel_initializer='normal', activation='relu',
	data_format='channels_last', input_shape=(train_x.shape[1], train_x.shape[2], train_x.shape[3])))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Convolution2D(filters=64, kernel_size=(4,4), kernel_initializer='normal', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(Convolution2D(filters=64, kernel_size=(4,4), kernel_initializer='normal', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(Flatten(data_format='channels_last'))
model.add(Dense(128, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

optimizer_ = 'Nadam'
loss_ = 'binary_crossentropy'
model.compile(optimizer=optimizer_, loss=loss_, metrics=['accuracy'])

es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=4)
classWeight = class_weight.compute_class_weight('balanced', np.unique(train_y), train_y[:])
batchSize = 256
numberOfEpochs = 30

model.fit(train_x, train_y,
		epochs = numberOfEpochs,
		batch_size = batchSize,
		class_weight = classWeight,
		callbacks = [es],
		validation_split = 0.2,
		shuffle = True)

savePath = workPath+'/models/trainedModels/'
model.save(savePath+'jetImages_model_'+databin+'.h5')

# Print final ROC AUC score
print ' - roc auc: ',round(roc_auc_score(test_y,model.predict(test_x)),3)

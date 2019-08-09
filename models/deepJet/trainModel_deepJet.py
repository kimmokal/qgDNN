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
trainSetPath = workPath+"/data/trainSets/deepJet/preprocessed_deepJet_trainSet_"+databin+".root"
testSetPath = workPath+"/data/testSets/deepJet/preprocessed_deepJet_testSet_"+databin+".root"

df_train = root_pandas.read_root(trainSetPath, 'tree')
df_test = root_pandas.read_root(testSetPath, 'tree')

# Extract the target labels
train_y = df_train.isPhysUDS.copy()
test_y = df_test.isPhysUDS.copy()

train_x = df_train
train_x.drop(['isPhysUDS'], axis=1, inplace=True)
test_x = df_test
test_x.drop(['isPhysUDS'], axis=1, inplace=True)

# Separate the particle flow (PF) and jet variables
cPFs = list(df_train.filter(regex='jetPFc'))
nPFs = list(df_train.filter(regex='jetPFn'))
cPF_train_x = train_x[cPFs]
cPF_test_x = test_x[cPFs]
nPF_train_x = train_x[nPFs]
nPF_test_x = test_x[nPFs]

cPF0 = list(cPF_train_x.filter(regex='0'))
cPF0_train_x = cPF_train_x[cPF0]
cPF0_test_x = cPF_test_x[cPF0]
cPF1 = list(cPF_train_x.filter(regex='1'))
cPF1_train_x = cPF_train_x[cPF1]
cPF1_test_x = cPF_test_x[cPF1]
cPF2 = list(cPF_train_x.filter(regex='2'))
cPF2_train_x = cPF_train_x[cPF2]
cPF2_test_x = cPF_test_x[cPF2]
cPF3 = list(cPF_train_x.filter(regex='3'))
cPF3_train_x = cPF_train_x[cPF3]
cPF3_test_x = cPF_test_x[cPF3]
cPF4 = list(cPF_train_x.filter(regex='4'))
cPF4_train_x = cPF_train_x[cPF4]
cPF4_test_x = cPF_test_x[cPF4]
cPF5 = list(cPF_train_x.filter(regex='5'))
cPF5_train_x = cPF_train_x[cPF5]
cPF5_test_x = cPF_test_x[cPF5]
cPF6 = list(cPF_train_x.filter(regex='6'))
cPF6_train_x = cPF_train_x[cPF6]
cPF6_test_x = cPF_test_x[cPF6]
cPF7 = list(cPF_train_x.filter(regex='7'))
cPF7_train_x = cPF_train_x[cPF7]
cPF7_test_x = cPF_test_x[cPF7]
cPF8 = list(cPF_train_x.filter(regex='8'))
cPF8_train_x = cPF_train_x[cPF8]
cPF8_test_x = cPF_test_x[cPF8]
cPF9 = list(cPF_train_x.filter(regex='9'))
cPF9_train_x = cPF_train_x[cPF9]
cPF9_test_x = cPF_test_x[cPF9]

cPF0_train_x = cPF0_train_x.as_matrix()
cPF0_test_x = cPF0_test_x.as_matrix()
cPF1_train_x = cPF1_train_x.as_matrix()
cPF1_test_x = cPF1_test_x.as_matrix()
cPF2_train_x = cPF2_train_x.as_matrix()
cPF2_test_x = cPF2_test_x.as_matrix()
cPF3_train_x = cPF3_train_x.as_matrix()
cPF3_test_x = cPF3_test_x.as_matrix()
cPF4_train_x = cPF4_train_x.as_matrix()
cPF4_test_x = cPF4_test_x.as_matrix()
cPF5_train_x = cPF5_train_x.as_matrix()
cPF5_test_x = cPF5_test_x.as_matrix()
cPF6_train_x = cPF6_train_x.as_matrix()
cPF6_test_x = cPF6_test_x.as_matrix()
cPF7_train_x = cPF7_train_x.as_matrix()
cPF7_test_x = cPF7_test_x.as_matrix()
cPF8_train_x = cPF8_train_x.as_matrix()
cPF8_test_x = cPF8_test_x.as_matrix()
cPF9_train_x = cPF9_train_x.as_matrix()
cPF9_test_x = cPF9_test_x.as_matrix()

cPFs_train_x = np.dstack([cPF0_train_x, cPF1_train_x, cPF2_train_x, cPF3_train_x, cPF4_train_x, cPF5_train_x,
			cPF6_train_x, cPF7_train_x, cPF8_train_x, cPF9_train_x])
cPFs_train_x = np.transpose(cPFs_train_x, (0, 2, 1))
cPFs_test_x = np.dstack([cPF0_test_x, cPF1_test_x, cPF2_test_x, cPF3_test_x, cPF4_test_x, cPF5_test_x,
			cPF6_test_x, cPF7_test_x, cPF8_test_x, cPF9_test_x])
cPFs_test_x = np.transpose(cPFs_test_x, (0, 2, 1))

nPF0 = list(nPF_train_x.filter(regex='0'))
nPF0_train_x = nPF_train_x[nPF0]
nPF0_test_x = nPF_test_x[nPF0]
nPF1 = list(nPF_train_x.filter(regex='1'))
nPF1_train_x = nPF_train_x[nPF1]
nPF1_test_x = nPF_test_x[nPF1]
nPF2 = list(nPF_train_x.filter(regex='2'))
nPF2_train_x = nPF_train_x[nPF2]
nPF2_test_x = nPF_test_x[nPF2]
nPF3 = list(nPF_train_x.filter(regex='3'))
nPF3_train_x = nPF_train_x[nPF3]
nPF3_test_x = nPF_test_x[nPF3]
nPF4 = list(nPF_train_x.filter(regex='4'))
nPF4_train_x = nPF_train_x[nPF4]
nPF4_test_x = nPF_test_x[nPF4]
nPF5 = list(nPF_train_x.filter(regex='5'))
nPF5_train_x = nPF_train_x[nPF5]
nPF5_test_x = nPF_test_x[nPF5]
nPF6 = list(nPF_train_x.filter(regex='6'))
nPF6_train_x = nPF_train_x[nPF6]
nPF6_test_x = nPF_test_x[nPF6]
nPF7 = list(nPF_train_x.filter(regex='7'))
nPF7_train_x = nPF_train_x[nPF7]
nPF7_test_x = nPF_test_x[nPF7]
nPF8 = list(nPF_train_x.filter(regex='8'))
nPF8_train_x = nPF_train_x[nPF8]
nPF8_test_x = nPF_test_x[nPF8]
nPF9 = list(nPF_train_x.filter(regex='9'))
nPF9_train_x = nPF_train_x[nPF9]
nPF9_test_x = nPF_test_x[nPF9]

nPF0_train_x = nPF0_train_x.as_matrix()
nPF0_test_x = nPF0_test_x.as_matrix()
nPF1_train_x = nPF1_train_x.as_matrix()
nPF1_test_x = nPF1_test_x.as_matrix()
nPF2_train_x = nPF2_train_x.as_matrix()
nPF2_test_x = nPF2_test_x.as_matrix()
nPF3_train_x = nPF3_train_x.as_matrix()
nPF3_test_x = nPF3_test_x.as_matrix()
nPF4_train_x = nPF4_train_x.as_matrix()
nPF4_test_x = nPF4_test_x.as_matrix()
nPF5_train_x = nPF5_train_x.as_matrix()
nPF5_test_x = nPF5_test_x.as_matrix()
nPF6_train_x = nPF6_train_x.as_matrix()
nPF6_test_x = nPF6_test_x.as_matrix()
nPF7_train_x = nPF7_train_x.as_matrix()
nPF7_test_x = nPF7_test_x.as_matrix()
nPF8_train_x = nPF8_train_x.as_matrix()
nPF8_test_x = nPF8_test_x.as_matrix()
nPF9_train_x = nPF9_train_x.as_matrix()
nPF9_test_x = nPF9_test_x.as_matrix()

nPFs_train_x = np.dstack([nPF0_train_x, nPF1_train_x, nPF2_train_x, nPF3_train_x, nPF4_train_x, nPF5_train_x,
			nPF6_train_x, nPF7_train_x, nPF8_train_x, nPF9_train_x])
nPFs_train_x = np.transpose(nPFs_train_x, (0, 2, 1))
nPFs_test_x = np.dstack([nPF0_test_x, nPF1_test_x, nPF2_test_x, nPF3_test_x, nPF4_test_x, nPF5_test_x,
			nPF6_test_x, nPF7_test_x, nPF8_test_x, nPF9_test_x])
nPFs_test_x = np.transpose(nPFs_test_x, (0, 2, 1))

PFs_train_x = np.concatenate((cPFs_train_x, nPFs_train_x), axis=1)
PFs_test_x = np.concatenate((cPFs_test_x, nPFs_test_x), axis=1)

jetVar_train_x = train_x.drop(cPFs+nPFs, axis=1)
jetVar_test_x = test_x.drop(cPFs+nPFs, axis=1)

jetVar_train_x = jetVar_train_x.as_matrix()
jetVar_test_x = jetVar_test_x.as_matrix()

# Build the neural network
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Convolution1D, LSTM, CuDNNLSTM, Activation, Dropout, Flatten, Concatenate, BatchNormalization
from keras import optimizers
from sklearn.utils import class_weight

PF_input = Input(shape=(20, 5), name='PF_input')
conv1 = Convolution1D(filters=64, kernel_size=1, kernel_initializer='normal', activation='relu')(PF_input)
bn1 = BatchNormalization()(conv1)
dropout1 = Dropout(0.1)(bn1)
conv2 = Convolution1D(filters=32, kernel_size=1, kernel_initializer='normal',  activation='relu')(dropout1)
bn2 = BatchNormalization()(conv2)
dropout2 = Dropout(0.1)(bn2)
conv3 = Convolution1D(filters=32, kernel_size=1, kernel_initializer='normal',  activation='relu')(dropout2)
bn3 = BatchNormalization()(conv3)
dropout3 = Dropout(0.1)(bn3)
conv4 = Convolution1D(filters=8, kernel_size=1, kernel_initializer='normal',  activation='relu')(dropout3)
rnn1 = LSTM(150, go_backwards=True)(conv4)
# rnn1 = CuDNNLSTM(150, go_backwards=True)(conv4)
bn_out = BatchNormalization()(rnn1)

# Input for global features
jetVar_input = Input(shape=(jetVar_train_x.shape[1],), name='jetVar_input')

merge = keras.layers.concatenate([jetVar_input, bn_out])
dense1 = Dense(200, activation='relu',kernel_initializer='normal')(merge)
bn4 = BatchNormalization()(dense1)
dropout4 = Dropout(0.1)(bn4)
dense2 =  Dense(100, activation='relu',kernel_initializer='normal')(dropout4)
bn5 = BatchNormalization()(dense2)
dropout5 = Dropout(0.1)(bn5)
dense3 =  Dense(100, activation='relu',kernel_initializer='normal')(dropout5)
network_output = Dense(1, activation='sigmoid', kernel_initializer='normal', name='network_output')(dense3)
deepModel = Model(inputs=[PF_input, jetVar_input], outputs=network_output)

optimizer_ = 'Nadam'
loss_ = 'binary_crossentropy'
deepModel.compile(optimizer=optimizer_, loss=loss_, metrics=['accuracy'])

# loss = LossHistory()
earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=4)
classWeight = class_weight.compute_class_weight('balanced', np.unique(train_y), train_y[:])
batchSize = 512
numberOfEpochs = 30

deepModel.fit(x={'PF_input': PFs_train_x, 'jetVar_input': jetVar_train_x}, y=train_y,
		epochs = numberOfEpochs,
		batch_size = batchSize,
		class_weight = classWeight,
		callbacks = [earlystop],
		validation_split = 0.2,
		shuffle = True)

# Save the trained model
savePath = workPath+'/models/trainedModels/'
deepModel.save(savePath+'deepJet_model_'+databin+'.h5')

# Print final ROC AUC score
print ' - roc auc: ',round(roc_auc_score(test_y,deepModel.predict({'PF_input': PFs_test_x, 'jetVar_input': jetVar_test_x})),3)

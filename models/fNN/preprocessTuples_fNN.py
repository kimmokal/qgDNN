import numpy as np
import pandas as pd
import root_numpy
import root_pandas

# Define the path to the working directory
workPath = "/work/kimmokal/qgDNN"

### Choose the eta, pT bin to be preprocessed
databin = "eta1.3_pt30to100"
# databin = "eta1.3_pt100to300"
# databin = "eta1.3_pt300to1000"
# databin = "eta1.3_pt1000"
# databin = "eta2.5_pt30to100"
# databin = "eta2.5_pt100to300"
# databin = "eta2.5_pt300to1000"

dataPath = workPath+"/data/binned/"
fQuarkPath = dataPath+"quark_"+databin+".root"
fGluonPath = dataPath+"gluon_"+databin+".root"

read = [u'jetQGl', u'QG_ptD', u'QG_axis2', u'QG_mult', u'isPhysUDS']

# Read in the files
df_quark = root_pandas.read_root(fQuarkPath, columns=read)
df_gluon = root_pandas.read_root(fGluonPath, columns=read)

# Split to train and test data sets
TEST_SET_SIZE_GLUON = 25000
TRAIN_SET_SIZE_GLUON = 125000
TEST_SET_SIZE_QUARK = 25000
TRAIN_SET_SIZE_QUARK = 125000

testSet_quark = df_quark.iloc[:TEST_SET_SIZE_QUARK, :]
trainSet_quark = df_quark.iloc[TEST_SET_SIZE_QUARK:TEST_SET_SIZE_QUARK+TRAIN_SET_SIZE_QUARK, :]

testSet_gluon = df_gluon.iloc[:TEST_SET_SIZE_GLUON, :]
trainSet_gluon = df_gluon.iloc[TEST_SET_SIZE_GLUON:TEST_SET_SIZE_GLUON+TRAIN_SET_SIZE_GLUON, :]

testSet = pd.concat([testSet_quark, testSet_gluon], ignore_index=True)
trainSet = pd.concat([trainSet_quark, trainSet_gluon], ignore_index=True)

# Shuffle the training data at this point
trainSet = trainSet.sample(frac=1).reset_index(drop=True)

# Normalize the dataset by substracting the mean from each column and dividing by the standard deviation
normCols = [col for col in trainSet.columns if col not in ['isPhysUDS', 'jetQGl']]

trainSet_mean = trainSet[normCols].mean()
trainSet_std = trainSet[normCols].std()

trainSet[normCols] = trainSet[normCols] - trainSet_mean
trainSet[normCols] = trainSet[normCols] / trainSet_std

testSet[normCols] = testSet[normCols] - trainSet_mean
testSet[normCols] = testSet[normCols] / trainSet_std

# Save the preprocessed sets
outFolderTest = workPath+"/data/testSets/fNN/"
outFolderTrain = workPath+"/data/trainSets/fNN/"

saveNameTest = outFolderTest+"preprocessed_FNN_testSet_"+databin+".root"
testSet.to_root(saveNameTest,key='tree')

saveNameTrain = outFolderTrain+"preprocessed_FNN_trainSet_"+databin+".root"
trainSet.to_root(saveNameTrain,key='tree')

print "Processed: ", databin

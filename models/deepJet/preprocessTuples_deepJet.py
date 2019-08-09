import numpy as np
import pandas as pd
import root_numpy
import root_pandas

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

read = [u'jetPt', u'jetEta', u'QG_ptD', u'QG_axis2', u'QG_mult', u'isPhysUDS']
PFcolumns = ['jetPF_pT', 'jetPF_pTrel', 'jetPF_dR', 'jetPF_id', 'jetPF_fromPV']
read=read+PFcolumns

#Function to flatten columns that contain lists as entries (for example Cpfcan_pt)
def flattencolumns(df1, cols,len):
	df = pd.concat([pd.DataFrame(df1[x].values.tolist()).add_prefix(x).iloc[:,:len] for x in cols], axis=1)
	df.fillna(0.0,inplace=True)
	df1.drop(cols, axis=1,inplace=True)
	df.reset_index(drop=True, inplace=True)
	df1.reset_index(drop=True, inplace=True)
	return pd.concat([df, df1], axis=1)

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

#How many particles are included per PF type
num_pfCands=10

# Separate charged and neutral PFs
chargedColumns =  ['jetPFc_pT', 'jetPFc_pTrel', 'jetPFc_dR', 'jetPFc_id', 'jetPFc_fromPV']
neutralColumns = ['jetPFn_pT', 'jetPFn_pTrel', 'jetPFn_dR', 'jetPFn_id', 'jetPFn_fromPV']
newColumns = chargedColumns + neutralColumns

for newCol in newColumns:
    trainSet[newCol] = [np.empty(0,dtype=float)]*len(trainSet)
    testSet[newCol] = [np.empty(0,dtype=float)]*len(testSet)

# Loop over the jets and separate the PFs
for row in trainSet.itertuples():
    chargedHadrons = np.abs(row.jetPF_id) == 211
    chargedElectrons = np.abs(row.jetPF_id) == 11
    chargedMuons = np.abs(row.jetPF_id) == 13
    chargedPFs = chargedHadrons+chargedElectrons+chargedMuons
    neutralPFs = np.invert(chargedPFs)
    trainSet.at[row.Index, 'jetPFc_pT'] = row.jetPF_pT[ chargedPFs ]
    trainSet.at[row.Index, 'jetPFc_pTrel'] = row.jetPF_pTrel[ chargedPFs ]
    trainSet.at[row.Index, 'jetPFc_dR'] = row.jetPF_dR[ chargedPFs ]
    trainSet.at[row.Index, 'jetPFc_id'] = row.jetPF_id[ chargedPFs ]
    trainSet.at[row.Index, 'jetPFc_fromPV'] = row.jetPF_fromPV[ chargedPFs ]
    trainSet.at[row.Index, 'jetPFn_pT'] = row.jetPF_pT[ neutralPFs ]
    trainSet.at[row.Index, 'jetPFn_pTrel'] = row.jetPF_pTrel[ neutralPFs ]
    trainSet.at[row.Index, 'jetPFn_dR'] = row.jetPF_dR[ neutralPFs ]
    trainSet.at[row.Index, 'jetPFn_id'] = row.jetPF_id[ neutralPFs ]
    trainSet.at[row.Index, 'jetPFn_fromPV'] = row.jetPF_fromPV[ neutralPFs ]

for row in testSet.itertuples():
    chargedHadrons = np.abs(row.jetPF_id) == 211
    chargedElectrons = np.abs(row.jetPF_id) == 11
    chargedMuons = np.abs(row.jetPF_id) == 13
    chargedPFs = chargedHadrons+chargedElectrons+chargedMuons
    neutralPFs = np.invert(chargedPFs)
    testSet.at[row.Index, 'jetPFc_pT'] = row.jetPF_pT[ chargedPFs ]
    testSet.at[row.Index, 'jetPFc_pTrel'] = row.jetPF_pTrel[ chargedPFs ]
    testSet.at[row.Index, 'jetPFc_dR'] = row.jetPF_dR[ chargedPFs ]
    testSet.at[row.Index, 'jetPFc_id'] = row.jetPF_id[ chargedPFs ]
    testSet.at[row.Index, 'jetPFc_fromPV'] = row.jetPF_fromPV[ chargedPFs ]
    testSet.at[row.Index, 'jetPFn_pT'] = row.jetPF_pT[ neutralPFs ]
    testSet.at[row.Index, 'jetPFn_pTrel'] = row.jetPF_pTrel[ neutralPFs ]
    testSet.at[row.Index, 'jetPFn_dR'] = row.jetPF_dR[ neutralPFs ]
    testSet.at[row.Index, 'jetPFn_id'] = row.jetPF_id[ neutralPFs ]
    testSet.at[row.Index, 'jetPFn_fromPV'] = row.jetPF_fromPV[ neutralPFs ]

trainSet.drop(PFcolumns, axis=1, inplace=True)
trainSet = flattencolumns(trainSet, newColumns, num_pfCands)

testSet.drop(PFcolumns, axis=1, inplace=True)
testSet = flattencolumns(testSet, newColumns, num_pfCands)

# Normalize the dataset by substracting the mean from each column and dividing by the standard deviation
normCols = [col for col in trainSet.columns if col not in ['isPhysUDS']]

trainSet_mean = trainSet[normCols].mean()
trainSet_std = trainSet[normCols].std()

trainSet[normCols] = trainSet[normCols] - trainSet_mean
trainSet[normCols] = trainSet[normCols] / trainSet_std

testSet[normCols] = testSet[normCols] - trainSet_mean
testSet[normCols] = testSet[normCols] / trainSet_std

# Save the preprocessed jets
outFolderTrain = workPath+"/data/trainSets/deepJet/"
outFolderTest = workPath+"/data/testSets/deepJet/"

saveNameTrain=outFolderTrain+"preprocessed_deepJet_trainSet_"+databin+".root"
trainSet.to_root(saveNameTrain,key='tree')

saveNameTest=outFolderTest+"preprocessed_deepJet_testSet_"+databin+".root"
testSet.to_root(saveNameTest,key='tree')

print "Processed: ", databin

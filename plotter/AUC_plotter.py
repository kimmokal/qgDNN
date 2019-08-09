# Restrict to one GPU
import imp
try:
        imp.find_module('setGPU')
        import setGPU
except ImportError:
        found = False

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import keras.backend as K
import pandas as pd
import numpy as np
import root_numpy
import keras.callbacks
import h5py

from sklearn.metrics import roc_auc_score, roc_curve, auc
from root_pandas import read_root

from keras.models import Sequential,Model,load_model
from keras.layers import Input,Dense,Convolution1D,Flatten,Dropout,Activation,LSTM,CuDNNLSTM,Concatenate,BatchNormalization, Convolution2D, MaxPooling2D
from keras.constraints import maxnorm
from keras import optimizers

workPath = "/work/kimmokal/qgDNN"
outFolder = workPath+"/plotter/plots/"

### Choose the eta, pT bin to be plotted
databin = "eta1.3_pt30to100"
binlabel = '$|\eta| < 1.3, \; \; \; 30$ GeV$ < p_T < 100$ GeV'

# databin = "eta1.3_pt100to300"
# binlabel = '$|\eta| < 1.3, \; \; \; 100$ GeV$ < p_T < 300$ GeV'

# databin = "eta1.3_pt300to1000"
# binlabel = '$|\eta| < 1.3, \; \; \; 300$ GeV$ < p_T < 1000$ GeV'

# databin = "eta1.3_pt1000"
# binlabel = '$|\eta| < 1.3, \; \; \; p_T > 1000$ GeV'

# databin = "eta2.5_pt30to100"
# binlabel = '$1.3 < |\eta| < 2.5, \; \; \; 30$ GeV$ < p_T < 100$ GeV'

# databin = "eta2.5_pt100to300"
# binlabel = '$1.3 < |\eta| < 2.5, \; \; \; 100$ GeV$ < p_T < 300$ GeV'

# databin = "eta2.5_pt300to1000"
# binlabel = '$1.3 < |\eta| < 2.5, \; \; \; 300$ GeV$ < p_T < 1000$ GeV'

#######################
###### FNN MODEL #######
#######################

fpath_FNN = workPath+"/data/testSets/fNN/preprocessed_FNN_testSet_"
df_FNN = read_root(fpath_FNN+databin+'.root', "tree")

test_FNN_y = df_FNN.isPhysUDS.copy()
test_FNN_x = df_FNN

test_FNN_x.drop(['isPhysUDS','jetQGl'], axis=1, inplace=True)

test_FNN_x = test_FNN_x.as_matrix()
test_FNN_y = test_FNN_y.as_matrix()

modelpath_FNN = workPath+"/models/trainedModels/FNN_model_"
FNN_model = load_model(modelpath_FNN+databin+".h5")

pred_FNN_y=FNN_model.predict(test_FNN_x)

#######################
###### LIKELIHOOD #######
#######################

fpath_QGl = workPath+"/data/testSets/fNN/preprocessed_FNN_testSet_"
df_QGl = read_root(fpath_QGl+databin+'.root', 'tree')

test_QGl_y = df_QGl.isPhysUDS.copy()

df_QGl.drop(['isPhysUDS','QG_mult', 'QG_axis2', 'QG_ptD'], axis=1, inplace=True)

# Extract the QG likelihood discriminator's predictions
# For jetQGl, a gluon jet is towards 0 and quark jet is towards 1, so it needs to be flipped
pred_QGl_y = df_QGl.jetQGl.copy()

test_QGl_y = test_QGl_y.as_matrix()
pred_QGl_y = pred_QGl_y.as_matrix()

#######################
######## DEEPJET ########
#######################

fpath_DJ = workPath+"/data/testSets/deepJet/preprocessed_deepJet_testSet_"
df_DJ = read_root(fpath_DJ+databin+'.root', 'tree')

test_DJ_y = df_DJ.isPhysUDS.copy()

test_DJ_x = df_DJ
test_DJ_x.drop(['isPhysUDS'], axis=1, inplace=True)

print(list(test_DJ_x))

cPFs = list(test_DJ_x.filter(regex='jetPFc'))
nPFs = list(test_DJ_x.filter(regex='jetPFn'))
cPF_test_x = test_DJ_x[cPFs]
nPF_test_x = test_DJ_x[nPFs]

cPF0 = list(cPF_test_x.filter(regex='0'))
cPF0_test_x = cPF_test_x[cPF0]
cPF1 = list(cPF_test_x.filter(regex='1'))
cPF1_test_x = cPF_test_x[cPF1]
cPF2 = list(cPF_test_x.filter(regex='2'))
cPF2_test_x = cPF_test_x[cPF2]
cPF3 = list(cPF_test_x.filter(regex='3'))
cPF3_test_x = cPF_test_x[cPF3]
cPF4 = list(cPF_test_x.filter(regex='4'))
cPF4_test_x = cPF_test_x[cPF4]
cPF5 = list(cPF_test_x.filter(regex='5'))
cPF5_test_x = cPF_test_x[cPF5]
cPF6 = list(cPF_test_x.filter(regex='6'))
cPF6_test_x = cPF_test_x[cPF6]
cPF7 = list(cPF_test_x.filter(regex='7'))
cPF7_test_x = cPF_test_x[cPF7]
cPF8 = list(cPF_test_x.filter(regex='8'))
cPF8_test_x = cPF_test_x[cPF8]
cPF9 = list(cPF_test_x.filter(regex='9'))
cPF9_test_x = cPF_test_x[cPF9]

cPF0_test_x = cPF0_test_x.as_matrix()
cPF1_test_x = cPF1_test_x.as_matrix()
cPF2_test_x = cPF2_test_x.as_matrix()
cPF3_test_x = cPF3_test_x.as_matrix()
cPF4_test_x = cPF4_test_x.as_matrix()
cPF5_test_x = cPF5_test_x.as_matrix()
cPF6_test_x = cPF6_test_x.as_matrix()
cPF7_test_x = cPF7_test_x.as_matrix()
cPF8_test_x = cPF8_test_x.as_matrix()
cPF9_test_x = cPF9_test_x.as_matrix()

nPF0 = list(nPF_test_x.filter(regex='0'))
nPF0_test_x = nPF_test_x[nPF0]
nPF1 = list(nPF_test_x.filter(regex='1'))
nPF1_test_x = nPF_test_x[nPF1]
nPF2 = list(nPF_test_x.filter(regex='2'))
nPF2_test_x = nPF_test_x[nPF2]
nPF3 = list(nPF_test_x.filter(regex='3'))
nPF3_test_x = nPF_test_x[nPF3]
nPF4 = list(nPF_test_x.filter(regex='4'))
nPF4_test_x = nPF_test_x[nPF4]
nPF5 = list(nPF_test_x.filter(regex='5'))
nPF5_test_x = nPF_test_x[nPF5]
nPF6 = list(nPF_test_x.filter(regex='6'))
nPF6_test_x = nPF_test_x[nPF6]
nPF7 = list(nPF_test_x.filter(regex='7'))
nPF7_test_x = nPF_test_x[nPF7]
nPF8 = list(nPF_test_x.filter(regex='8'))
nPF8_test_x = nPF_test_x[nPF8]
nPF9 = list(nPF_test_x.filter(regex='9'))
nPF9_test_x = nPF_test_x[nPF9]

nPF0_test_x = nPF0_test_x.as_matrix()
nPF1_test_x = nPF1_test_x.as_matrix()
nPF2_test_x = nPF2_test_x.as_matrix()
nPF3_test_x = nPF3_test_x.as_matrix()
nPF4_test_x = nPF4_test_x.as_matrix()
nPF5_test_x = nPF5_test_x.as_matrix()
nPF6_test_x = nPF6_test_x.as_matrix()
nPF7_test_x = nPF7_test_x.as_matrix()
nPF8_test_x = nPF8_test_x.as_matrix()
nPF9_test_x = nPF9_test_x.as_matrix()

nPFs_test_x = np.dstack([nPF0_test_x, nPF1_test_x, nPF2_test_x, nPF3_test_x, nPF4_test_x, nPF5_test_x,
			nPF6_test_x, nPF7_test_x, nPF8_test_x, nPF9_test_x])
nPFs_test_x = np.transpose(nPFs_test_x, (0, 2, 1))

cPFs_test_x = np.dstack([cPF0_test_x, cPF1_test_x, cPF2_test_x, cPF3_test_x, cPF4_test_x, cPF5_test_x,
			cPF6_test_x, cPF7_test_x, cPF8_test_x, cPF9_test_x])
cPFs_test_x = np.transpose(cPFs_test_x, (0, 2, 1))

PFs_test_x = np.concatenate((cPFs_test_x, nPFs_test_x), axis=1)

jetVar_test_x = test_DJ_x
jetVar_test_x.drop(cPFs+nPFs, axis=1, inplace=True)
jetVar_test_x = jetVar_test_x.as_matrix()

modelpath_DJ = workPath+"/trainedModels/deepJet_model_"
DJ_model = load_model(modelpath_DJ+databin+".h5")

pred_DJ_y = DJ_model.predict(x=[PFs_test_x, jetVar_test_x])
test_DJ_y=test_DJ_y.as_matrix()

#######################
####### JET IMAGE #######
#######################

fpath_Img = workPath+"/data/testSets/jetImages/"
fname_gluon = "jetImagesGluon_testSet_"
fname_quark = "jetImagesQuark_testSet_"

h5g = h5py.File(fpath_Img + fname_gluon + databin + '.h5', 'r')
gluonNtPt = h5g['NtPt'][()]
gluonChPt = h5g['ChPt'][()]
gluonChMult = h5g['ChMult'][()]
h5q = h5py.File(fpath_Img + fname_quark + databin + '.h5', 'r')
quarkNtPt = h5q['NtPt'][()]
quarkChPt = h5q['ChPt'][()]
quarkChMult = h5q['ChMult'][()]

test_Img_Quark_y = np.empty(quarkNtPt.shape[0], dtype=int); test_Img_Quark_y.fill(1)
test_Img_Gluon_y = np.empty(gluonNtPt.shape[0], dtype=int); test_Img_Gluon_y.fill(0)

test_NtPt = np.concatenate((gluonNtPt, quarkNtPt), axis=0)
test_ChPt = np.concatenate((gluonChPt, quarkChPt), axis=0)
test_ChMult = np.concatenate((gluonChMult, quarkChMult), axis=0)

test_Img_x = np.stack((test_NtPt, test_ChPt, test_ChMult), axis=-1)
test_Img_y = np.concatenate((test_Img_Gluon_y, test_Img_Quark_y))

modelpath_Img = workPath+"/models/trainedModels/jetImages_model_"
Img_model = load_model(modelpath_Img+databin+".h5")
pred_Img_y = Img_model.predict(test_Img_x)

#### PLOT ROC CURVE ######
print ' - FNN roc auc: ',round(roc_auc_score(test_FNN_y,pred_FNN_y),3)
print ' - QGl roc auc: ', round(roc_auc_score(test_QGl_y, pred_QGl_y),3)
print ' - DJ roc auc: ',round(roc_auc_score(test_DJ_y,pred_DJ_y),3)
print ' - Jet images roc auc: ',round(roc_auc_score(test_Img_y,pred_Img_y),3)

fpr_FNN, tpr_FNN, thresholds_FNN  = roc_curve(test_FNN_y, pred_FNN_y)
roc_auc_FNN = auc(fpr_FNN, tpr_FNN)

fpr_QGl, tpr_QGl, thresholds_QGl  = roc_curve(test_QGl_y, pred_QGl_y)
roc_auc_QGl = auc(fpr_QGl, tpr_QGl)

fpr_DJ, tpr_DJ, thresholds_DJ  = roc_curve(test_DJ_y, pred_DJ_y)
roc_auc_DJ = auc(fpr_DJ, tpr_DJ)

fpr_Img, tpr_Img, thresholds_Img  = roc_curve(test_Img_y, pred_Img_y)
roc_auc_Img = auc(fpr_Img, tpr_Img)

plt.clf()
plt.figure(figsize=(5,5))
plt.plot(fpr_QGl, tpr_QGl, 'r', label='Likelihood AUC = %0.3f'% roc_auc_QGl)
plt.plot(fpr_FNN, tpr_FNN, 'b', label='Feedforward AUC = %0.3f'% roc_auc_FNN)
plt.plot(fpr_DJ, tpr_DJ, 'forestgreen', label='Sequential Model AUC = %0.3f'% roc_auc_DJ)
plt.plot(fpr_Img, tpr_Img, 'purple', label='Jet Image AUC = %0.3f'% roc_auc_Img)
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.legend(loc='lower right')
plt.title(binlabel)
plt.ylabel('Quark jet acceptance rate')
plt.xlabel('Gluon jet acceptance rate')
plt.savefig(outFolder + 'roc_curve_' + databin + '.pdf')

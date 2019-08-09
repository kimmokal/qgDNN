import numpy as np
import pandas as pd
import root_numpy
import root_pandas
import h5py
import time

from matplotlib import pyplot as plt
import matplotlib.colors as colors

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

read = [u'isPhysUDS', u'isPhysG']
PFcolumns = ['PF_pT', 'PF_dR', 'PF_dTheta', 'PF_id', 'PF_fromPV']

read = read + PFcolumns

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

print "Initiate image processing!"
start_time = time.time()

# Add two new columns for xy-coordinates
testSet['PF_x'] = testSet.PF_pT
testSet['PF_y'] = testSet.PF_pT

trainSet['PF_x'] = trainSet.PF_pT
trainSet['PF_y'] = trainSet.PF_pT

print "Step 1, time elapsed: ", (time.time()-start_time)

# Drop PF candidates with fromPV = 0,1 and sort the rest by pT
for row in trainSet.itertuples():
	notPU = row.PF_fromPV > 1
	notOutOfRange = row.PF_dR < 1.0
	trainSet.at[row.Index, 'PF_pT'] = row.PF_pT[ notPU & notOutOfRange ]
	trainSet.at[row.Index, 'PF_dR'] = row.PF_dR[ notPU & notOutOfRange ]
	trainSet.at[row.Index, 'PF_dTheta'] = row.PF_dTheta[ notPU & notOutOfRange ]
	trainSet.at[row.Index, 'PF_id'] = row.PF_id[ notPU & notOutOfRange ]
	trainSet.at[row.Index, 'PF_fromPV'] = row.PF_fromPV[ notPU & notOutOfRange ]
	trainSet.at[row.Index, 'PF_x'] = row.PF_x[ notPU & notOutOfRange ]
	trainSet.at[row.Index, 'PF_y'] = row.PF_y[ notPU & notOutOfRange ]

for row in testSet.itertuples():
	notPU = row.PF_fromPV > 1
	notOutOfRange = row.PF_dR < 1.0
	testSet.at[row.Index, 'PF_pT'] = row.PF_pT[ notPU & notOutOfRange ]
	testSet.at[row.Index, 'PF_dR'] = row.PF_dR[ notPU & notOutOfRange ]
	testSet.at[row.Index, 'PF_dTheta'] = row.PF_dTheta[ notPU & notOutOfRange ]
	testSet.at[row.Index, 'PF_id'] = row.PF_id[ notPU & notOutOfRange ]
	testSet.at[row.Index, 'PF_fromPV'] = row.PF_fromPV[ notPU & notOutOfRange ]
	testSet.at[row.Index, 'PF_x'] = row.PF_x[ notPU & notOutOfRange ]
	testSet.at[row.Index, 'PF_y'] = row.PF_y[ notPU & notOutOfRange ]

print "Step 2, time elapsed: ", (time.time()-start_time)

for row in trainSet.itertuples():
	sortPermutation = row.PF_pT.argsort()
	sortPermutation = np.flipud(sortPermutation)
	trainSet.at[row.Index, 'PF_pT'] = row.PF_pT[ sortPermutation ]
	trainSet.at[row.Index, 'PF_dR'] = row.PF_dR[ sortPermutation ]
	trainSet.at[row.Index, 'PF_dTheta'] = row.PF_dTheta[ sortPermutation ]
	trainSet.at[row.Index, 'PF_id'] = row.PF_id[ sortPermutation ]
	trainSet.at[row.Index, 'PF_fromPV'] = row.PF_fromPV[ sortPermutation ]

for row in testSet.itertuples():
	sortPermutation = row.PF_pT.argsort()
	sortPermutation = np.flipud(sortPermutation)
	testSet.at[row.Index, 'PF_pT'] = row.PF_pT[ sortPermutation ]
	testSet.at[row.Index, 'PF_dR'] = row.PF_dR[ sortPermutation ]
	testSet.at[row.Index, 'PF_dTheta'] = row.PF_dTheta[ sortPermutation ]
	testSet.at[row.Index, 'PF_id'] = row.PF_id[ sortPermutation ]
	testSet.at[row.Index, 'PF_fromPV'] = row.PF_fromPV[ sortPermutation ]

print "Step 3, time elapsed: ", (time.time()-start_time)

for row in trainSet.itertuples():
	trainSet.at[row.Index, 'PF_x'] = np.multiply(row.PF_dR, np.cos(row.PF_dTheta))
	trainSet.at[row.Index, 'PF_y'] = np.multiply(row.PF_dR, np.sin(row.PF_dTheta))

for row in testSet.itertuples():
	testSet.at[row.Index, 'PF_x'] = np.multiply(row.PF_dR, np.cos(row.PF_dTheta))
	testSet.at[row.Index, 'PF_y'] = np.multiply(row.PF_dR, np.sin(row.PF_dTheta))

# Jet images
imageSize = 33
edgeSize = 2
# Choose which variables to project as images
trainSet_gluonNtPt = []
trainSet_gluonChPt = []
trainSet_gluonChMult = []
trainSet_quarkNtPt = []
trainSet_quarkChPt = []
trainSet_quarkChMult = []

testSet_gluonNtPt = []
testSet_gluonChPt = []
testSet_gluonChMult = []
testSet_quarkNtPt = []
testSet_quarkChPt = []
testSet_quarkChMult = []

# Translate highest pT particle to the center, (x,y) = (0,0)
def imageTranslation( dataset, dataframe, dataframe_Idx ):
	dataset.at[ dataframe_Idx, 'PF_x'] = dataframe.PF_x - dataframe.PF_x[0]
	dataset.at[ dataframe_Idx, 'PF_y'] = dataframe.PF_y - dataframe.PF_y[0]
	dataset.at[ dataframe_Idx, 'PF_dR'] = np.sqrt(dataset.at[ dataframe_Idx, 'PF_x']**2 + dataset.at[ dataframe_Idx, 'PF_y']**2)
	dataset.at[ dataframe_Idx, 'PF_dTheta'] = np.arctan2(dataset.at[ dataframe_Idx, 'PF_y'], dataset.at[ dataframe_Idx, 'PF_x'])

# Rotate the image so that the particle with second highest pT is at theta=0
def imageRotation( dataset, dataframe, dataframe_Idx ):
	dataset.at[ dataframe_Idx, 'PF_dTheta'] = dataframe.PF_dTheta - dataframe.PF_dTheta[1]
	dataset.at[ dataframe_Idx, 'PF_x'] = np.multiply(dataframe.PF_dR, np.cos(dataset.at[ dataframe_Idx, 'PF_dTheta'] ))
	dataset.at[ dataframe_Idx, 'PF_y'] = np.multiply(dataframe.PF_dR, np.sin(dataset.at[ dataframe_Idx, 'PF_dTheta'] ))

# Flip the image with respect to the y-axis if needed
def imageFlip( dataset, dataframe, dataframe_Idx ):
	weightedpT = np.sum( np.multiply(dataframe.PF_pT, dataframe.PF_y) )
	if (weightedpT < 0):
		dataset.at[ dataframe_Idx, 'PF_y'] = dataframe.PF_y * (-1)
		dataset.at[ dataframe_Idx, 'PF_dTheta'] = dataframe.PF_dTheta + np.pi

# Do the preprocessing steps for each jet
for row in trainSet.itertuples():
	imageTranslation( trainSet, row, row.Index )

for row in testSet.itertuples():
	imageTranslation( testSet, row, row.Index )

print "Step 4, time elapsed: ", (time.time()-start_time)

# for row in df.itertuples():
# # 	imageRotationEllipse( row, row.Index )
# 	imageRotation( row, row.Index )

print "Step 5, time elapsed: ", (time.time()-start_time)

for row in trainSet.itertuples():
	imageFlip( trainSet, row, row.Index )

for row in testSet.itertuples():
	imageFlip( testSet, row, row.Index )

print "Step 6, time elapsed: ", (time.time()-start_time)

# Pixelize the jets
for row in trainSet.itertuples():
	trainSet.at[ row.Index, 'PF_x' ] = ((row.PF_x+1.0)*0.5*imageSize).astype(int) + edgeSize/2
	trainSet.at[ row.Index, 'PF_y' ] = ((row.PF_y+1.0)*0.5*imageSize).astype(int) + edgeSize/2

for row in testSet.itertuples():
	testSet.at[ row.Index, 'PF_x' ] = ((row.PF_x+1.0)*0.5*imageSize).astype(int) + edgeSize/2
	testSet.at[ row.Index, 'PF_y' ] = ((row.PF_y+1.0)*0.5*imageSize).astype(int) + edgeSize/2

print "Step 7, time elapsed: ", (time.time()-start_time)

for row in trainSet.itertuples():
	# Create empty images and fill them with jet images
	imgNtPt = np.zeros((imageSize+edgeSize, imageSize+edgeSize), dtype=np.float32)
	imgChPt = np.zeros((imageSize+edgeSize, imageSize+edgeSize),  dtype=np.float32)
	imgChMult = np.zeros((imageSize+edgeSize, imageSize+edgeSize),  dtype=np.float32)
	for i in range(row.PF_x.size):
		pf_x = trainSet.at[ row.Index, 'PF_x' ][i]
		pf_y = trainSet.at[ row.Index, 'PF_y' ][i]
		if ( np.abs(pf_x) < imageSize+edgeSize/2 and np.abs(pf_y) < imageSize+edgeSize/2):
			if ((np.abs(row.PF_id[i]) == 211) or (np.abs(row.PF_id[i]) == 11) or (np.abs(row.PF_id[i]) == 13)):
				imgChPt[pf_x, pf_y] += row.PF_pT[i]
				imgChMult[pf_x, pf_y] += 1
			else:
				imgNtPt[pf_x, pf_y] += row.PF_pT[i]
	# Normalize the images; minAdd prevents division by zero
	minAdd = 0.000001
	imgNtPt = imgNtPt / (np.linalg.norm(imgNtPt) + minAdd)
	imgChPt = imgChPt / (np.linalg.norm(imgChPt) + minAdd)
	imgChMult = imgChMult / (np.linalg.norm(imgChMult) + minAdd)
	# Add the images of the jet to the jet image collections
	# Check whether its a gluon or quark jet and then check if the particle is charged
	if (row.isPhysG == 1):
		trainSet_gluonNtPt.append( imgNtPt )
		trainSet_gluonChPt.append( imgChPt )
		trainSet_gluonChMult.append( imgChMult )
	else:
		trainSet_quarkNtPt.append( imgNtPt )
		trainSet_quarkChPt.append( imgChPt )
		trainSet_quarkChMult.append( imgChMult )

for row in testSet.itertuples():
	# Create empty images and fill them with jet images
	imgNtPt = np.zeros((imageSize+edgeSize, imageSize+edgeSize), dtype=np.float32)
	imgChPt = np.zeros((imageSize+edgeSize, imageSize+edgeSize),  dtype=np.float32)
	imgChMult = np.zeros((imageSize+edgeSize, imageSize+edgeSize),  dtype=np.float32)
	for i in range(row.PF_x.size):
		pf_x = testSet.at[ row.Index, 'PF_x' ][i]
		pf_y = testSet.at[ row.Index, 'PF_y' ][i]
		if ( np.abs(pf_x) < imageSize+edgeSize/2 and np.abs(pf_y) < imageSize+edgeSize/2):
			if ((np.abs(row.PF_id[i]) == 211) or (np.abs(row.PF_id[i]) == 11) or (np.abs(row.PF_id[i]) == 13)):
				imgChPt[pf_x, pf_y] += row.PF_pT[i]
				imgChMult[pf_x, pf_y] += 1
			else:
				imgNtPt[pf_x, pf_y] += row.PF_pT[i]
	# Normalize the images; minAdd prevents division by zero
	minAdd = 0.000001
	imgNtPt = imgNtPt / (np.linalg.norm(imgNtPt) + minAdd)
	imgChPt = imgChPt / (np.linalg.norm(imgChPt) + minAdd)
	imgChMult = imgChMult / (np.linalg.norm(imgChMult) + minAdd)
	# Add the images of the jet to the jet image collections
	# Check whether its a gluon or quark jet and then check if the particle is charged
	if (row.isPhysG == 1):
		testSet_gluonNtPt.append( imgNtPt )
		testSet_gluonChPt.append( imgChPt )
		testSet_gluonChMult.append( imgChMult )
	else:
		testSet_quarkNtPt.append( imgNtPt )
		testSet_quarkChPt.append( imgChPt )
		testSet_quarkChMult.append( imgChMult )

print "Step 8, time elapsed: ", (time.time()-start_time)

# Calculate the mean (average) and standard deviation of the trainSet images
avgTrainGluonNtPtImage = np.sum(trainSet_gluonNtPt, axis=0) / len(trainSet_gluonNtPt)
avgTrainGluonChPtImage = np.sum(trainSet_gluonChPt, axis=0) / len(trainSet_gluonChPt)
avgTrainGluonChMultImage = np.sum(trainSet_gluonChMult, axis=0) / len(trainSet_gluonChMult)

avgTrainQuarkNtPtImage = np.sum(trainSet_quarkNtPt, axis=0) / len(trainSet_quarkNtPt)
avgTrainQuarkChPtImage = np.sum(trainSet_quarkChPt, axis=0) / len(trainSet_quarkChPt)
avgTrainQuarkChMultImage = np.sum(trainSet_quarkChMult, axis=0) / len(trainSet_quarkChMult)

stdTrainGluonNtPtImage = np.std(trainSet_gluonNtPt, axis=0)
stdTrainGluonChPtImage = np.std(trainSet_gluonChPt, axis=0)
stdTrainGluonChMultImage = np.std(trainSet_gluonChMult, axis=0)

stdTrainQuarkNtPtImage = np.std(trainSet_quarkNtPt, axis=0)
stdTrainQuarkChPtImage = np.std(trainSet_quarkChPt, axis=0)
stdTrainQuarkChMultImage = np.std(trainSet_quarkChMult, axis=0)

# Calculate the mean (average) and standard deviation of ALL the images
allTrainNtPt = np.concatenate((trainSet_quarkNtPt, trainSet_gluonNtPt), axis=0)
allTrainChPt = np.concatenate((trainSet_quarkChPt, trainSet_gluonChPt), axis=0)
allTrainChMult = np.concatenate((trainSet_quarkChMult, trainSet_gluonChMult), axis=0)

avgTrainNtPtImage = np.sum(allTrainNtPt, axis=0) / len(allTrainNtPt)
avgTrainChPtImage = np.sum(allTrainChPt, axis=0) / len(allTrainChPt)
avgTrainChMultImage = np.sum(allTrainChMult, axis=0) / len(allTrainChMult)

stdTrainNtPtImage = np.std(allTrainNtPt, axis=0)
stdTrainChPtImage = np.std(allTrainChPt, axis=0)
stdTrainChMultImage = np.std(allTrainChMult, axis=0)

# Zero-center the images by subtracting the average image
trainSet_gluonNtPt = trainSet_gluonNtPt - avgTrainNtPtImage
trainSet_gluonChPt = trainSet_gluonChPt - avgTrainChPtImage
trainSet_gluonChMult = trainSet_gluonChMult - avgTrainChMultImage

trainSet_quarkNtPt = trainSet_quarkNtPt - avgTrainNtPtImage
trainSet_quarkChPt = trainSet_quarkChPt - avgTrainChPtImage
trainSet_quarkChMult = trainSet_quarkChMult - avgTrainChMultImage

testSet_gluonNtPt = testSet_gluonNtPt - avgTrainNtPtImage
testSet_gluonChPt = testSet_gluonChPt - avgTrainChPtImage
testSet_gluonChMult = testSet_gluonChMult - avgTrainChMultImage

testSet_quarkNtPt = testSet_quarkNtPt - avgTrainNtPtImage
testSet_quarkChPt = testSet_quarkChPt - avgTrainChPtImage
testSet_quarkChMult = testSet_quarkChMult - avgTrainChMultImage

# Divide by the standard deviation
trainSet_gluonNtPt = np.true_divide(trainSet_gluonNtPt, (stdTrainNtPtImage + 0.00001))
trainSet_gluonChPt = np.true_divide(trainSet_gluonChPt, (stdTrainChPtImage + 0.00001))
trainSet_gluonChMult = np.true_divide(trainSet_gluonChMult, (stdTrainChMultImage + 0.00001))

trainSet_quarkNtPt = np.true_divide(trainSet_quarkNtPt, (stdTrainNtPtImage + 0.00001))
trainSet_quarkChPt = np.true_divide(trainSet_quarkChPt, (stdTrainChPtImage + 0.00001))
trainSet_quarkChMult = np.true_divide(trainSet_quarkChMult, (stdTrainChMultImage + 0.00001))

testSet_gluonNtPt = np.true_divide(testSet_gluonNtPt, (stdTrainNtPtImage + 0.00001))
testSet_gluonChPt = np.true_divide(testSet_gluonChPt, (stdTrainChPtImage + 0.00001))
testSet_gluonChMult = np.true_divide(testSet_gluonChMult, (stdTrainChMultImage + 0.00001))

testSet_quarkNtPt = np.true_divide(testSet_quarkNtPt, (stdTrainNtPtImage + 0.00001))
testSet_quarkChPt = np.true_divide(testSet_quarkChPt, (stdTrainChPtImage + 0.00001))
testSet_quarkChMult = np.true_divide(testSet_quarkChMult, (stdTrainChMultImage + 0.00001))

# Scale the average images for illustrative purposes
showAvgGluonNtPtImage = np.sqrt(avgTrainGluonNtPtImage)
showAvgGluonChPtImage = np.sqrt(avgTrainGluonChPtImage)
showAvgGluonChMultImage = np.sqrt(avgTrainGluonChMultImage)

showAvgQuarkNtPtImage = np.sqrt(avgTrainQuarkNtPtImage)
showAvgQuarkChPtImage = np.sqrt(avgTrainQuarkChPtImage)
showAvgQuarkChMultImage = np.sqrt(avgTrainQuarkChMultImage)

print "Step 9, time elapsed: ", (time.time()-start_time)

outFolderTrain = workPath+"/data/trainSets/jetImages/"
outFolderTest = workPath+"/data/testSets/jetImages/"

# Save the jets to h5 files
saveNameGluonTrain = outFolderTrain + "jetImagesGluon_trainSet_"+ databin +".h5"
with h5py.File(saveNameGluonTrain, 'w') as h5g:
	h5g.create_dataset('NtPt', data=trainSet_gluonNtPt)
	h5g.create_dataset('ChPt', data=trainSet_gluonChPt)
	h5g.create_dataset('ChMult', data=trainSet_gluonChMult)

saveNameQuarkTrain = outFolderTrain + "jetImagesQuark_trainSet_"+ databin +".h5"
with h5py.File(saveNameQuarkTrain, 'w') as h5q:
	h5q.create_dataset('NtPt', data=trainSet_quarkNtPt)
	h5q.create_dataset('ChPt', data=trainSet_quarkChPt)
	h5q.create_dataset('ChMult', data=trainSet_quarkChMult)

saveNameGluonTest = outFolderTest + "jetImagesGluon_testSet_"+ databin +".h5"
with h5py.File(saveNameGluonTest, 'w') as h5g:
	h5g.create_dataset('NtPt', data=testSet_gluonNtPt)
	h5g.create_dataset('ChPt', data=testSet_gluonChPt)
	h5g.create_dataset('ChMult', data=testSet_gluonChMult)

saveNameQuarkTest = outFolderTest + "jetImagesQuark_testSet_"+ databin +".h5"
with h5py.File(saveNameQuarkTest, 'w') as h5q:
	h5q.create_dataset('NtPt', data=testSet_quarkNtPt)
	h5q.create_dataset('ChPt', data=testSet_quarkChPt)
	h5q.create_dataset('ChMult', data=testSet_quarkChMult)

print "Processed:", databin, ". Time elapsed: ", (time.time()-start_time)

# # Save the averages plot
# fig, ((ax1,ax2,ax3), (ax4,ax5,ax6)) = plt.subplots(2,3)
# fig.suptitle('Q/G jet images')
# fig.set_size_inches(10., 6.7)
# ax1.set_title('Gluon neutral PF pT')
# im1 = ax1.imshow(showAvgGluonNtPtImage)
# ax1.axis('off')
# ax2.set_title('Gluon charged PF pT')
# im2 = ax2.imshow(showAvgGluonChPtImage )
# ax2.axis('off')
# ax3.set_title('Gluon charged PF multiplicity')
# im3 = ax3.imshow(showAvgGluonChMultImage)
# ax3.axis('off')
# ax4.set_title('Quark neutral PF pT')
# im4 = ax4.imshow(showAvgQuarkNtPtImage)
# ax4.axis('off')
# ax5.set_title('Quark charged PF pT')
# im5 = ax5.imshow(showAvgQuarkChPtImage)
# ax5.axis('off')
# ax6.set_title('Quark charged PF multiplicity')
# im6 = ax6.imshow(showAvgQuarkChMultImage)
# ax6.axis('off')
# plt.savefig("average_images_"+databin+".png")
#
# # Save the differences plot
# fig2, (ax1_diff, ax2_diff, ax3_diff) = plt.subplots(1,3)
# fig2.set_size_inches(10., 6.7)
# ax1_diff.set_title('Neutral pT')
# im1_diff = ax1_diff.imshow(showAvgQuarkNtPtImage - showAvgGluonNtPtImage, cmap='bwr')
# ax1_diff.axis('off')
# ax2_diff.set_title('Charged pT')
# im2_diff = ax2_diff.imshow(showAvgQuarkChPtImage - showAvgGluonChPtImage, cmap='bwr')
# ax2_diff.axis('off')
# ax3_diff.set_title('Charged multiplicity')
# im3_diff = ax3_diff.imshow(showAvgQuarkChMultImage - showAvgGluonChMultImage, cmap='bwr')
# ax3_diff.axis('off')
# plt.savefig("difference_images_"+databin+".png")
#
# # Save the differences plot
# fig3, (ax1_sqrt, ax2_sqrt, ax3_sqrt) = plt.subplots(1,3)
# fig3.set_size_inches(10., 6.7)
# ax1_sqrt.set_title('Neutral pT')
# im1_sqrt = ax1_sqrt.imshow(showAvgQuarkNtPtImage - showAvgGluonNtPtImage, cmap='bwr')
# ax1_sqrt.axis('off')
# ax2_sqrt.set_title('Charged pT')
# im2_sqrt = ax2_sqrt.imshow(showAvgQuarkChPtImage - showAvgGluonChPtImage, cmap='bwr')
# ax2_sqrt.axis('off')
# ax3_sqrt.set_title('Charged multiplicity')
# im3_sqrt = ax3_sqrt.imshow(showAvgQuarkChMultImage - showAvgGluonChMultImage, cmap='bwr')
# ax3_sqrt.axis('off')
# plt.savefig("difference_images_sqrt_"+databin+".png")

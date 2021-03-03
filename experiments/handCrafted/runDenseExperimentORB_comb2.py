import os
import sys
import numpy
import pickle
import sklearn.cluster
import sklearn.ensemble

cvPath = "/scratch/georgioutk/opencv/install/lib/python3.6/dist-packages"
sys.path.append(cvPath)
clusteringPath = "/scratch/georgioutk"
sys.path.append(clusteringPath)
import cv2
import clustering
import preprocessing
# from matplotlib import pyplot as plt


nJobs=88
descriptor = "ORB/"
print("Loading Data")
try:
	trainSet = numpy.load("trainSetInUint8.npy")
	trainLabels = numpy.load(preprocessing.trainLabelPath).astype(numpy.float32)
except FileNotFoundError:
	trainSet, trainLabels = preprocessing.loadTrainSet()
	numpy.save("trainSetInUint8.npy", trainSet)

try:
	testSet = numpy.load("testSetInUint8.npy")
	testLabels = numpy.load(preprocessing.testLabelPath).astype(numpy.float32)
except FileNotFoundError:
	testSet, testLabels = preprocessing.loadTestSet()
	numpy.save("testSetInUint8.npy", testSet)

# surf = cv2.BRISK_create()
surf = cv2.ORB_create()
surf.setFastThreshold(0)
surf.setEdgeThreshold(0)
# surf = cv2.xfeatures2d.SURF_create()
# surf = cv2.xfeatures2d.SIFT_create()
# surf = cv2.xfeatures2d.FREAK_create()
# surf = cv2.xfeatures2d.BriefDescriptorExtractor_create()

print("Extracting train features")
try:
	print("Loading")
	descriptors = pickle.load(open(descriptor+"denseTrainDescriptors.pkl", "rb"))
	# descriptors = numpy.concatenate(descriptors, axis=0)
except FileNotFoundError:
	count = 0
	descriptors = []
	for im in trainSet:
		count += 1
		if count % 100 == 0:
			print(count/15000, end="\r", flush=True)
		kp = []
		for size in [12, 16, 24, 32]:
			for i in range(0, im.shape[0], size):
				for j in range(0, im.shape[1], size):
					kp.append(cv2.KeyPoint(i,j,_size=size))
		descs = []
		for i in range(5):
			d = surf.compute(im[:,:,i], kp)[1]
			descs.append(d)
		# descs = numpy.concatenate(descs, axis=1)
		descriptors.append(descs)
	# descriptors = numpy.concatenate(descriptors, axis=0)
	pickle.dump(descriptors, open(descriptor+"denseTrainDescriptors.pkl", "wb"))

print("Combining modalities")
for c, d in enumerate(descriptors):
	descriptors[c] = numpy.concatenate(d, axis=1)

nWords = 512
print("Creating dictionary")
allDescs = numpy.concatenate(descriptors, axis=0)
if allDescs.shape[0]>1e5:
	allDescs = allDescs[:int(1e5)]

try:
	dictionary = pickle.load(open(descriptor+"denseCombinedMyDictionary_"+str(nWords)+".pkl", "rb"))
except FileNotFoundError:
	km = clustering.KMedians.KMedians(n_clusters=nWords, nJobs=nJobs)
	# km = clustering.KMeans.ApproximateKMeans(n_clusters=nWords, nJobs=nJobs)
	dictionary = km.fit(allDescs)
	pickle.dump(dictionary, open(descriptor+"denseCombinedMyDictionary_"+str(nWords)+".pkl", "wb"))
	# km = sklearn.cluster.KMeans(n_clusters=nWords, n_init=1, max_iter=3000, n_jobs=64)
	# dictionary = km.fit(allDescs)
	# pickle.dump(dictionary, open(descriptor+"combinedMyDictionary_"+str(nWords)+".pkl", "wb"))

del allDescs

#Version 1
# print("Extracting train descriptors")
# wrTrain = numpy.zeros([trainSet.shape[0], nWords])
# count = 0
# for im in trainSet:
# 	if count % 100 == 0:
# 		print(count/15000, end="\r", flush=True)
# 	kp = det.detect(im[:,:,0])
# 	for i in range(1,5):
# 		nkp = det.detect(im[:,:,i])
# 		toGo = []
# 		for k in range(len(nkp)):
# 			for kk in kp:
# 				if kk.overlap(kk, nkp[k]) > 0.9:
# 					toGo.append(k)
# 					break
# 		for k in toGo[::-1]:
# 			del(nkp[k])
# 		kp += nkp
# 	descs = []
# 	if len(kp) == 0: continue
# 	for i in range(5):
# 		descs.append(surf.compute(im[:,:,i], kp)[1])
# 	descs = numpy.concatenate(descs, axis=1)
# 	wIndexes = dictionary.predict(descs)
# 	wrTrain[count, wIndexes] += 1
# 	count += 1

#Version 2
# print("Extracting train descriptors")
# wrTrain = numpy.zeros([trainSet.shape[0], nWords])
# count = 0
# for im in descriptors:
# 	if count % 100 == 0:
# 		print(count/15000, end="\r", flush=True)
# 	wIndexes = dictionary.predict(im)
# 	un, unCount = numpy.unique(wIndexes, return_counts=True)
# 	wrTrain[count, un] += unCount
# 	# wrTrain[count, wIndexes] += 1
# 	count += 1


#Version3
print("Extracting train descriptors")
wrTrain = numpy.zeros([trainSet.shape[0], nWords])
count = 0
numDescs = descriptors[0].shape[0]
descriptors = numpy.concatenate(descriptors, axis=0)
print('descriptors shape')
print(descriptors.shape)
if descriptors.shape[0] > 250000:
	tempDescs = numpy.array_split(descriptors, descriptors.shape[0]//250000)
else:
	tempDescs = descriptors
wIndexes = []
for tDescs in tempDescs:
	wIndexes.append(dictionary.predict(tDescs))
# wIndexes = dictionary.predict(descriptors)
wIndexes = numpy.concatenate(wIndexes, axis=0)
wIndexes = numpy.reshape(wIndexes, [trainSet.shape[0], numDescs])
for im in range(wIndexes.shape[0]):
	un, unCount = numpy.unique(wIndexes[im], return_counts=True)
	wrTrain[im, un] += unCount
	# wrTrain[count, wIndexes] += 1
	count += 1

print("Train to tf.idf")
summTrain = numpy.sum(wrTrain, axis=1, keepdims=True)
summTrain[numpy.where(summTrain==0)] = 1
tfTrain = wrTrain/summTrain
wSumTrain = numpy.sum(wrTrain, axis=0, keepdims=True)
wSumTrain[numpy.where(wSumTrain==0)] = 1
idf = numpy.log(wrTrain.shape[0]/wSumTrain)
tfIdfTrain = tfTrain*idf

# tfTrain = wrTrain/numpy.sum(wrTrain, axis=1, keepdims=True)
# idf = numpy.log(wrTrain.shape[0]/numpy.sum(wrTrain, axis=0, keepdims=True))
# tfIdfTrain = tfTrain*idf

print("Extracting test descriptors")
wrTest = numpy.zeros([testSet.shape[0], nWords])
count = 0
for im in testSet:
	if count % 100 == 0:
		print(count/1000, end="\r", flush=True)
	kp = []
	for size in [12, 16, 24, 32]:
		for i in range(0, im.shape[0], size):
			for j in range(0, im.shape[1], size):
				kp.append(cv2.KeyPoint(i,j,_size=size))
	descs = []
	if len(kp) == 0:
		exit("Test example with no keypoints")
	for i in range(5):
		descs.append(surf.compute(im[:,:,i], kp)[1])
	descs = numpy.concatenate(descs, axis=1)
	wIndexes = dictionary.predict(descs)
	un, unCount = numpy.unique(wIndexes, return_counts=True)
	wrTest[count, un] += unCount
	# wrTest[count, wIndexes] += 1
	count += 1

print("Test to tf.idf")
tfTest = wrTest/numpy.sum(wrTest, axis=1, keepdims=True)
tfIdfTest = tfTest*idf

print("Training Random Forests")
rf = sklearn.ensemble.RandomForestRegressor(n_jobs=60)
rf.fit(tfIdfTrain, trainLabels)

print("Predicting")
preds = rf.predict(tfIdfTest)

diff = numpy.square(preds-testLabels)
mean = numpy.sqrt(numpy.mean(diff, axis=0))

log = open(descriptor+"V3denseCombinedMy_"+str(nWords)+".acc", "w")
print("Performance")
print(mean, file=log)


# count/15000

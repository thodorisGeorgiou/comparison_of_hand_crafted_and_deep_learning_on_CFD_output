import os
import sys
import numpy
import time
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


nJobs=64
trainSetSize = 15000
descriptor = "ORB/"
# descriptor = "trainSetSize_2000/"
print("Loading Data")
st = time.time()
try:
	trainSet = numpy.load("trainSetInUint8.npy")[:trainSetSize]
	trainLabels = numpy.load(preprocessing.trainLabelPath).astype(numpy.float32)[:trainSetSize]
except FileNotFoundError:
	trainSet, trainLabels = preprocessing.loadTrainSet()
	trainSet = trainSet[:trainSetSize]
	trainLabels = trainLabels[:trainSetSize]
	# numpy.save("trainSetInUint8.npy", trainSet)

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

nWords = 256
print("Creating dictionary")

try:
	dictionaries = pickle.load(open(descriptor+"densePerModalityMyDictionaries_"+str(nWords)+".pkl", "rb"))
except FileNotFoundError:
	dictionaries = {}
	for i in range(5):
		print(i, end="\r", flush=True)
		allDescs = []
		for c, d in enumerate(descriptors):
			allDescs.append(d[i])
		allDescs = numpy.concatenate(allDescs, axis=0)
		if allDescs.shape[0]>1e5:
			allDescs = allDescs[:int(1e5)]
		# km = clustering.KMeans.ApproximateKMeans(n_clusters=nWords, nJobs=nJobs)
		km = clustering.KMedians.KMedians(n_clusters=nWords, nJobs=nJobs)
		# km = clustering.KMeans.ApproximateKMeans(n_clusters=nWords, nJobs=nJobs)
		dictionaries[i] = km.fit(allDescs)
	pickle.dump(dictionaries, open(descriptor+"densePerModalityMyDictionaries_"+str(nWords)+".pkl", "wb"))
	del allDescs
	# km = sklearn.cluster.KMeans(n_clusters=nWords, n_init=1, max_iter=3000, n_jobs=64)
	# dictionary = km.fit(allDescs)
	# pickle.dump(dictionary, open(descriptor+"combinedMyDictionary_"+str(nWords)+".pkl", "wb"))


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
# wrTrain = numpy.zeros([trainSet.shape[0], 5*nWords])
# count = 0
# for im in descriptors:
# 	if count % 100 == 0:
# 		print(count/15000, end="\r", flush=True)
# 	pWords = 0
# 	for mod in range(5):
# 		wIndexes = dictionaries[mod].predict(im[mod]) + pWords
# 		wrTrain[count, wIndexes] += 1
# 		pWords += nWords
# 	count += 1

#Version 3
print("Extracting train descriptors")
wrTrain = numpy.zeros([trainSet.shape[0], 5*nWords])
allDescs = [list([]), list([]), list([]), list([]), list([])]
for im in descriptors:
	for mod in range(5):
		allDescs[mod].append(im[mod])

pWords = 0
numDescs = descriptors[0][0].shape[0]
print(numDescs)
for mod in range(5):
	allDescs[mod] = numpy.concatenate(allDescs[mod], axis=0)
	print(allDescs[mod].shape)
	if allDescs[mod].shape[0] > 500000:
		tempDescs = numpy.array_split(allDescs[mod], allDescs[mod].shape[0]//500000)
	else:
		tempDescs = allDescs[mod]
	wIndexes = []
	for tDescs in tempDescs:
		wIndexes.append(dictionaries[mod].predict(tDescs) + pWords)
	wIndexes = numpy.concatenate(wIndexes, axis=0)
	wIndexes = numpy.reshape(wIndexes, [trainSet.shape[0], numDescs])
	for im in range(trainSetSize):
		un, unCount = numpy.unique(wIndexes[im], return_counts=True)
		wrTrain[im, un] += unCount
	pWords += nWords

print("Train to tf.idf")
summTrain = numpy.sum(wrTrain, axis=1, keepdims=True)
summTrain[numpy.where(summTrain==0)] = 1
tfTrain = wrTrain/summTrain
wSumTrain = numpy.sum(wrTrain, axis=0, keepdims=True)
wSumTrain[numpy.where(wSumTrain==0)] = 1
idf = numpy.log(wrTrain.shape[0]/wSumTrain)
tfIdfTrain = tfTrain*idf

print("Training Random Forests")
rf = sklearn.ensemble.RandomForestRegressor(n_jobs=60)
rf.fit(tfIdfTrain, trainLabels)
et = time.time()
print("Time to extract and train: "+str(et-st))
# tfTrain = wrTrain/numpy.sum(wrTrain, axis=1, keepdims=True)
# idf = numpy.log(wrTrain.shape[0]/numpy.sum(wrTrain, axis=0, keepdims=True))
# tfIdfTrain = tfTrain*idf

print("Extracting test descriptors")
wrTest = numpy.zeros([testSet.shape[0], 5*nWords])
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
	pWords = 0
	for i in range(5):
		descs = surf.compute(im[:,:,i], kp)[1]
		wIndexes = dictionaries[i].predict(descs) + pWords
		un, unCount = numpy.unique(wIndexes, return_counts=True)
		wrTest[count, un] += unCount
		pWords += nWords
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

log = open(descriptor+"V3densePerModalityMy_"+str(nWords)+".acc", "w")
print("Performance")
print(mean, file=log)


# count/15000

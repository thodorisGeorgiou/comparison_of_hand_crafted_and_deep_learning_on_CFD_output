import os
import sys
import numpy
import pickle
import sklearn.cluster
import sklearn.ensemble
import sklearn.metrics

cvPath = "/scratch/georgioutk/opencv/install/lib/python3.6/dist-packages"
clusteringPath = "/scratch/georgioutk"
sys.path.append(cvPath)
sys.path.append(clusteringPath)
import cv2
import preprocessing
import clustering

nJobs=64
descriptor = "ORB_AGAST/"
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

surf = cv2.ORB_create()
surf.setFastThreshold(0)
surf.setEdgeThreshold(0)
# surf = cv2.xfeatures2d.BriefDescriptorExtractor_create()
# surf = cv2.xfeatures2d.SURF_create()
# surf = cv2.xfeatures2d.SIFT_create()
# det = cv2.xfeatures2d.StarDetector_create()
# det = cv2.xfeatures2d.SURF_create()
# det = cv2.xfeatures2d.SIFT_create()
det = cv2.AgastFeatureDetector_create()
#det = cv2.ORB_create()
#det.setFastThreshold(0)
#det.setEdgeThreshold(0)

print("Extracting train features")
try:
	descriptors = pickle.load(open(descriptor+"allTrainDescriptors.pkl", "rb"))
except FileNotFoundError:
	descriptors = {}
	count = 0
	for im in trainSet:
		count += 1
		if count % 100 == 0:
			print(count/15000, end="\r", flush=True)
		for i in range(5):
			kp = det.detect(im[:,:,i])
			if len(kp)>0:
				print("In at least once")
				descs = surf.compute(im[:,:,i], kp)[1]
				try:
					descriptors[i] = numpy.concatenate([descriptors[i], descs], axis=0)
				except (KeyError, ValueError) as e:
					if type(e) == ValueError:
						pass
					descriptors[i] = descs
	pickle.dump(descriptors, open(descriptor+"allTrainDescriptors.pkl", "wb"))


nWords = [64,64,64,1024,1024]
totalWords = 0
for nw in nWords:
	totalWords += nw

print("Creating dictionaries")
try:
	dictionaries = pickle.load(open(descriptor+"perModalityMyDictionaries_"+str(nWords[0])+"_"+str(nWords[-1])+".pkl", "rb"))
except FileNotFoundError:
	dictionaries = {}
	for i in range(5):
		print(i, end="\r", flush=True)
		# km = sklearn.cluster.KMeans(n_clusters=nWords[i], n_init=1, max_iter=3000, n_jobs=60)
		# dictionaries[i] = clustering.KMeans.ApproximateKMeans(n_clusters=nWords[i], nJobs=nJobs)
		dictionaries[i] = clustering.KMedians.KMedians(n_clusters=nWords[i], nJobs=nJobs)
		try:
			if descriptors[i].shape[0] > 1e5:
				_descriptors = descriptors[i][:int(1e5)]
			else:
				_descriptors = descriptors[i]
			dictionaries[i].fit(_descriptors)
		except KeyError:
			pass
	pickle.dump(dictionaries, open(descriptor+"perModalityMyDictionaries_"+str(nWords[0])+"_"+str(nWords[-1])+".pkl", "wb"))


print("Extracting train descriptors")
wrTrain = numpy.zeros([trainSet.shape[0], totalWords])
count = 0
for im in trainSet:
	if count % 100 == 0:
		print(count/15000, end="\r", flush=True)
	pWords = 0
	for i in range(5):
		kp = det.detect(im[:,:,i])
		if len(kp)>0:
			descs = surf.compute(im[:,:,i], kp)[1]
			try:
				wIndexes = dictionaries[i].predict(descs) + pWords
			except KeyError:
				print("Something is very wrong!!!")
				continue
			un, unCount = numpy.unique(wIndexes, return_counts=True)
			wrTrain[count, un] += unCount
		pWords += nWords[i]
	count += 1

print("Train to tf.idf")
summTrain = numpy.sum(wrTrain, axis=1, keepdims=True)
summTrain[numpy.where(summTrain==0)] = 1
tfTrain = wrTrain/summTrain
wSumTrain = numpy.sum(wrTrain, axis=0, keepdims=True)
wSumTrain[numpy.where(wSumTrain==0)] = 1
idf = numpy.log(wrTrain.shape[0]/wSumTrain)
tfIdfTrain = tfTrain*idf

print("Extracting test descriptors")
wrTest = numpy.zeros([testSet.shape[0], totalWords])
count = 0
for im in testSet:
	if count % 100 == 0:
		print(count/1000, end="\r", flush=True)
	pWords = 0
	for i in range(5):
		kp = det.detect(im[:,:,i])
		if len(kp)>0:
			descs = surf.compute(im[:,:,i], kp)[1]
			try:
				wIndexes = dictionaries[i].predict(descs) + pWords
			except KeyError:
				continue
			un, unCount = numpy.unique(wIndexes, return_counts=True)
			wrTest[count, un] += unCount
			# wrTest[count, wIndexes] += 1
		pWords += nWords[i]
	count += 1


print("Test to tf.idf")
summTest = numpy.sum(wrTest, axis=1, keepdims=True)
summTest[numpy.where(summTest==0)] = 1
tfTest = wrTest/summTest
tfIdfTest = tfTest*idf

print("Training Random Forests")
rf = sklearn.ensemble.RandomForestRegressor(n_jobs=60)
rf.fit(tfIdfTrain, trainLabels)

print("Predicting")
preds = rf.predict(tfIdfTest)

diff = numpy.square(preds-testLabels)
print("Performance")
mean = numpy.sqrt(numpy.mean(diff, axis=0))
# print(mean)

log = open(descriptor+"V3separate_"+str(nWords[0])+"_"+str(nWords[-1])+".acc", "w")
print("Performance")
print(mean, file=log)

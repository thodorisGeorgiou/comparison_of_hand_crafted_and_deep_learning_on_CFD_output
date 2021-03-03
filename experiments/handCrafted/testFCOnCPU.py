import os
import sys
import numpy
import time
import pickle
import sklearn.cluster
import sklearn.ensemble

sys.path.append("/tank/georgioutk/cliffordConvolution/")
import cliffordConvolution as cc
cvPath = "/scratch/georgioutk/opencv/install/lib/python3.6/dist-packages"
sys.path.append(cvPath)
clusteringPath = "/scratch/georgioutk"
sys.path.append(clusteringPath)
import cv2
import clustering
import preprocessing
# from matplotlib import pyplot as plt

# monitorOutput = sys.stdout
# rubishOutput = open(os.devnull, "w")
# sys.stdout = rubishOutput
# sys.stderr = rubishOutput
# sys.stdwar = rubishOutput
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# sys.path.append("/tank/georgioutk/cliffordConvolution/")
import modelsFullSkip as models

batch_size = 100
MOVING_AVERAGE_DECAY = 0.999
INITIAL_LEARNING_RATE = 1e-3

nJobs=60
trainSetSize = 15000
# trainSetSize = int(sys.argv[1])
descriptor = "SIFT/"
# descriptor = "KMeansTrainSetSize_"+sys.argv[1]+"/"

# train_dir = os.getcwd() + "/" + descriptor + "mlp/"+run
train_dir = sys.argv[1]

def ema_to_weights(ema, variables):
	return tf.group(*(tf.assign(var, ema.average(var).read_value()) for var in variables))

def to_testing(ema):
	return ema_to_weights(ema, model_vars)

def testNetwork(sess, loss, testBatch_size, iterator):
	sess.run(iterator.initializer)
	count = 0
	mean = 0
	std = 0
	while True:
		try:
			res = sess.run(loss)
			mean += res
			std += res**2
			count += 1
		except tf.errors.OutOfRangeError:
			break
	return numpy.sqrt(mean/count), std/count

if __name__ == '__main__':
	print("Loading Data")
	st = time.time()
	print("Loading Train")
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

	print("Loading Val")
	try:
		valSet = numpy.load("valSetInUint8.npy")
		valLabels = numpy.load(preprocessing.valLabelPath).astype(numpy.float32)
	except FileNotFoundError:
		valSet, valLabels = preprocessing.loadValSet()
		numpy.save("valSetInUint8.npy", valSet)

	# surf = cv2.ORB_create()
	# surf = cv2.BRISK_create()
	# surf.setFastThreshold(0)
	# surf.setEdgeThreshold(0)
	# surf = cv2.xfeatures2d.SURF_create()
	surf = cv2.xfeatures2d.SIFT_create()
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
			# km = clustering.KMedians.KMedians(n_clusters=nWords, nJobs=nJobs)
			km = sklearn.cluster.KMeans(n_clusters=nWords, n_init=1, max_iter=3000, n_jobs=nJobs)
			# km = clustering.KMeans.ApproximateKMeans(n_clusters=nWords, nJobs=nJobs)
			dictionaries[i] = km.fit(allDescs)
		pickle.dump(dictionaries, open(descriptor+"densePerModalityMyDictionaries_"+str(nWords)+".pkl", "wb"))
		del allDescs
		# dictionary = km.fit(allDescs)
		# pickle.dump(dictionary, open(descriptor+"combinedMyDictionary_"+str(nWords)+".pkl", "wb"))

	#Version 3
	print("Extracting train descriptors")
	try:
		tfIdfTrain = numpy.load(descriptor+"PerModality_tfIdfTrain_"+str(nWords)+".npy")
		idf = numpy.load(descriptor+"PerModality_idf_"+str(nWords)+".npy")
	except FileNotFoundError:
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
			wIndexes = dictionaries[mod].predict(allDescs[mod]) + pWords
			print(wIndexes.shape)
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

		numpy.save(descriptor+"PerModality_idf_"+str(nWords), idf)
		numpy.save(descriptor+"PerModality_tfIdfTrain_"+str(nWords), tfIdfTrain)

	print("Extracting validation descriptors")
	try:
		tfIdfVal = numpy.load(descriptor+"PerModality_tfIdfVal_"+str(nWords)+".npy")
	except FileNotFoundError:
		wrVal = numpy.zeros([valSet.shape[0], 5*nWords])
		count = 0
		for im in valSet:
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
				wrVal[count, un] += unCount
				pWords += nWords
			count += 1

		print("Validation to tf.idf")
		tfVal = wrVal/numpy.sum(wrVal, axis=1, keepdims=True)
		tfIdfVal = tfVal*idf
		numpy.save(descriptor+"PerModality_tfIdfVal_"+str(nWords), tfIdfVal)

	print("Extracting test descriptors")
	try:
		tfIdfTest = numpy.load(descriptor+"PerModality_tfIdfTest_"+str(nWords)+".npy")
	except FileNotFoundError:
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

		numpy.save(descriptor+"PerModality_tfIdfTest_"+str(nWords), tfIdfTest)

	log = open(train_dir+"Testing.txt", "w", 1)
	global_step = tf.Variable(0, trainable=False)

	testDataset = tf.data.Dataset.from_tensor_slices((tfIdfTest.astype(numpy.float32), testLabels))
	testDataset = testDataset.batch(batch_size)
	testIterator = testDataset.make_initializable_iterator()
	testCode, batchTestLabels = testIterator.get_next()
	testPredictions = models.predictForces(testCode, batch_size, log, useType="test", first=True)

	testError = tf.reduce_mean(tf.square(tf.subtract(testPredictions, batchTestLabels)), axis=0)
	print("Towers defined.")

	# Track the moving averages of all trainable variables.
	model_vars = tf.trainable_variables()
	variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	variables_averages_op = variable_averages.apply(model_vars)

	to_test_op = to_testing(variable_averages)

	saver = tf.train.Saver(tf.global_variables())
	saverMax = tf.train.Saver(tf.global_variables())

	myconfig = tf.ConfigProto(log_device_placement=False)
	sess = tf.Session(config=myconfig)

	ckpt = tf.train.get_checkpoint_state(train_dir)
	if ckpt and ckpt.model_checkpoint_path:
		# Restores from checkpoint
		print("Model path:\n{}".format(ckpt.model_checkpoint_path))
		saver.restore(sess, ckpt.model_checkpoint_path)

	sess.run(to_test_op)
	meanError, stdError = testNetwork(sess, testError, batch_size, testIterator)

	print(meanError, end=" - ")
	print(stdError)
	print(meanError, end=" - ", file=log)
	print(stdError, file=log)

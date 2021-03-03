#Old script, not working. Here for reference
import sys
import os
# sys.stdout = open(os.devnull, "w")
# sys.stderr = open(os.devnull, "w")
# sys.stdwar = open(os.devnull, "w")
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy
import tensorflow as tf
sys.path.append("/tank/georgioutk/cliffordConvolutionMoreTest2/")
import cliffordConvolution as cc

import preprocessing
import models

# import warnings
# tf.logging.set_verbosity(tf.logging.ERROR)

numGpus = 4
batch_size = 200
MOVING_AVERAGE_DECAY = 0.9999

# train_dir = os.getcwd()+"/"+sys.argv[1]
train_dir = sys.argv[1]
modelType = sys.argv[2]

def testNetwork(sess, loss, predictions, testBatch_size, iterator):
	sess.run(iterator.initializer)
	count = 0
	mean = 0
	std = 0
	while True:
		try:
			res, preds = sess.run([loss, predictions])
			mean += res
			std += res**2
			count += 1
			try:
				numpy.concatenate([allPredictions, preds], axis=0)
			except NameError:
				allPredictions = preds
		except tf.errors.OutOfRangeError:
			break
	return numpy.sqrt(mean/count), std/count, allPredictions

# with warnings.catch_warnings():
	# warnings.simplefilter("ignore")
log = open(train_dir+"Accuracy4GPUHBS_RF.txt", "w", 1)
valData, numValExamples, valIterator = preprocessing.inputValFlows(batch_size, preprocessing.valSetPath, preprocessing.valLabelPath)
trainData, numTrainExamples, trainIterator = preprocessing.inputValFlows(batch_size, preprocessing.trainSetPath, preprocessing.trainLabelPath)
# trainData, numTrainExamples, trainIterator = preprocessing.inputValFlows(batch_size, preprocessing.trainSetPath, preprocessing.trainLabelPath)
perGPUValData = [list([]) for i in range(numGpus)]
for tD in valData[:-1]:
	split = tf.split(tD, numGpus, axis=0)
	for gpu in range(numGpus):
		perGPUValData[gpu].append(split[gpu])

perGPUTrainData = [list([]) for i in range(numGpus)]
for tD in trainData[:-1]:
	split = tf.split(tD, numGpus, axis=0)
	for gpu in range(numGpus):
		perGPUTrainData[gpu].append(split[gpu])

netOut = []
for gpu in range(numGpus):
	with tf.name_scope('tower_%d' % (gpu)) as scope:
		with tf.device('/gpu:%d' % gpu):
			print(perGPUValData[gpu][0].get_shape())
			print(len(perGPUValData[gpu]))
			valCode = models.inference(perGPUValData[gpu], first=(gpu==0), useType="test", modelType=modelType)
			print(valCode.get_shape())
			gpuValPredictions = models.predictForces(valCode, 5*batch_size//numGpus, log, useType="test", first=(gpu==0))
			netOut.append(gpuValPredictions)

trainNetOut = []
for gpu in range(numGpus):
	with tf.name_scope('tower_%d' % (gpu)) as scope:
		with tf.device('/gpu:%d' % gpu):
			print(perGPUTrainData[gpu][0].get_shape())
			print(len(perGPUTrainData[gpu]))
			trainCode = models.inference(perGPUTrainData[gpu], first=False, useType="test", modelType=modelType)
			print(trainCode.get_shape())
			gpuTrainPredictions = models.predictForces(trainCode, 5*batch_size//numGpus, log, useType="test", first=(gpu==0))
			trainNetOut.append(gpuTrainPredictions)


fcCodes = tf.get_collection("fcCodes")
sess.run(trainIterator.initializer)
while True:
	try:
		res, preds = sess.run([loss, predictions])
		mean += res
		std += res**2
		count += 1
		try:
			numpy.concatenate([allPredictions, preds], axis=0)
		except NameError:
			allPredictions = preds
	except tf.errors.OutOfRangeError:
		break


diff = tf.subtract(valPredictions, valData[-1])
valError = tf.reduce_mean(tf.square(diff), axis=0)
print("Val towers defined.")

variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
variables_to_restore = variable_averages.variables_to_restore()
saver = tf.train.Saver(variables_to_restore)

myconfig = tf.ConfigProto()
myconfig.gpu_options.allow_growth = True
sess = tf.Session(config=myconfig)

ckpt = tf.train.get_checkpoint_state(train_dir)
if ckpt and ckpt.model_checkpoint_path:
	# Restores from checkpoint
	print("Model path:\n{}".format(ckpt.model_checkpoint_path))
	saver.restore(sess, ckpt.model_checkpoint_path)

meanError, stdError, allPredictions = testNetwork(sess, valError, allOutputs, batch_size, valIterator)

# print("Training Random Forests")
# rf = sklearn.ensemble.RandomForestRegressor(n_jobs=60)
# rf.fit(allPredictions, trainLabels)

# print("Predicting")
# preds = rf.predict(tfIdfTest)

# diff = numpy.square(preds-testLabels)
# mean = numpy.sqrt(numpy.mean(diff, axis=0))

# print("Performance")
# print(mean)



sys.stdout = sys.__stdout__
print(sys.argv[1], end=": ")
print(meanError, end=" - ")
print(stdError)
print(meanError, end=" - ", file=log)
print(stdError, file=log)

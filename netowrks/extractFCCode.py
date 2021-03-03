import sys
import os
# sys.stdout = open(os.devnull, "w")
# sys.stderr = open(os.devnull, "w")
# sys.stdwar = open(os.devnull, "w")
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy
import tensorflow as tf
sys.path.append("/tank/georgioutk/cliffordConvolution/")
import cliffordConvolution as cc

import preprocessing
import modelsFullSkip as models

# import warnings
# tf.logging.set_verbosity(tf.logging.ERROR)

numGpus = 1
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
log = open("extractingFC2Codes_2FC_test.txt", "w", 1)
valData, numValExamples, valIterator = preprocessing.inputValFlows(batch_size, preprocessing.testSetPath, preprocessing.testLabelPath)
# trainData, valData, numTrainExamples, numValExamples, valIterator = preprocessing.inputFlows(batch_size)
perGPUValData = [list([]) for i in range(numGpus)]
for tD in valData[:-1]:
	split = tf.split(tD, numGpus, axis=0)
	for gpu in range(numGpus):
		perGPUValData[gpu].append(split[gpu])

netOut = []
for gpu in range(numGpus):
	with tf.name_scope('tower_%d' % (gpu)) as scope:
		with tf.device('/gpu:%d' % gpu):
			print(perGPUValData[gpu][0].get_shape())
			print(len(perGPUValData[gpu]))
			valCode = models.inference(perGPUValData[gpu], first=(gpu==0), useType="test", modelType=modelType)
			print(valCode.get_shape())
			gpuValPredictions = models.predictForces(valCode, batch_size//numGpus, log, useType="test", first=(gpu==0))
			netOut.append(gpuValPredictions)

fcCodes = tf.get_collection("fc2Code")

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

print("Initializign queue")
sess.run(valIterator.initializer)
numpyFCCodes = []
print("Running")
while True:
	try:
		numpyFCCodes.append(sess.run(fcCodes)[0])
	except tf.errors.OutOfRangeError:
		break

numpyFCCodes = numpy.concatenate(numpyFCCodes, axis=0)
print("FC codes shape: ",end="")
print(numpyFCCodes.shape)
numpy.save("testFC2Codes_2FC", numpyFCCodes)
exit(0)


diff = tf.subtract(valPredictions, valData[-1])
valError = tf.reduce_mean(tf.square(diff), axis=0)
print("Val towers defined.")

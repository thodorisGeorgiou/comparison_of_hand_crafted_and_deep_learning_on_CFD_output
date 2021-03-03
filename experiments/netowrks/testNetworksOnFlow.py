#Test singular network on flow
import sys
import os
sys.stdout = open(os.devnull, "w")
sys.stderr = open(os.devnull, "w")
sys.stdwar = open(os.devnull, "w")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy
import tensorflow as tf
# sys.path.append("/tank/georgioutk/cliffordConvolutionMoreTest2/")
sys.path.append("/tank/georgioutk/cliffordConvolution/")
import cliffordConvolution as cc

import preprocessing
import modelsFullSkip as models
# import models

# import warnings
# tf.logging.set_verbosity(tf.logging.ERROR)

numGpus = 1
batch_size = 1
MOVING_AVERAGE_DECAY = 0.9999

# train_dir = os.getcwd()+"/"+sys.argv[1]
train_dir = sys.argv[1]
if train_dir[-1] != "/":
	exit("Train path must end in /")

modelType = sys.argv[2]

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
	return mean/count, std/count

global_step = tf.Variable(0, trainable=False)
# with warnings.catch_warnings():
	# warnings.simplefilter("ignore")
log = open(train_dir+"testPerformance.log", "w", 1)
valData, numValExamples, valIterator = preprocessing.inputValFlowsForFlowPrediction(batch_size, preprocessing.testSetPath)
# trainData, valData, numTrainExamples, numValExamples, valIterator = preprocessing.inputFlows(batch_size)
perGPUTestData = [list([]) for i in range(numGpus)]
perGPUTestLabels = [list([]) for i in range(numGpus)]
for tD in valData:
	gpuSplits = tf.split(tD, numGpus, axis=0)
	for gpu, gpuSplit in enumerate(gpuSplits):
		split = tf.split(gpuSplit, 2, axis=1)
		perGPUTestData[gpu].append(split[0])
		perGPUTestLabels[gpu].append(split[1])

for gpu in range(numGpus):
	perGPUTestLabels[gpu] = tf.concat(perGPUTestLabels[gpu], axis=-1)

testLabels = tf.concat(perGPUTestLabels, axis=0)
netOut = []
for gpu in range(numGpus):
	with tf.name_scope('tower_%d' % (gpu)) as scope:
		with tf.device('/gpu:%d' % gpu):
			print(perGPUTestData[gpu][0].get_shape())
			print(len(perGPUTestData[gpu]))
			valCode = models.inference(perGPUTestData[gpu], first=(gpu==0), useType="test", modelType=modelType)
			print(valCode.get_shape())
			gpuValPredictions = models.predictFlow(valCode, batch_size//numGpus, log, useType="test", first=(gpu==0))
			netOut.append(gpuValPredictions)

valPredictions = tf.concat(netOut, axis=0)
diff = tf.subtract(valPredictions, testLabels)
valError = tf.reduce_mean(tf.square(diff))
print("Val towers defined.")

# variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
# variables_to_restore = variable_averages.variables_to_restore()
# saver = tf.train.Saver(variables_to_restore)
# Track the moving averages of all trainable variables.
model_vars = tf.trainable_variables()
variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
variables_averages_op = variable_averages.apply(model_vars)
saver = tf.train.Saver(tf.global_variables())

global_variables = tf.global_variables()

myconfig = tf.ConfigProto()
myconfig.gpu_options.allow_growth = True
sess = tf.Session(config=myconfig)

ckpt = tf.train.get_checkpoint_state(train_dir)
if ckpt and ckpt.model_checkpoint_path:
	# Restores from checkpoint
	print("Model path:\n{}".format(ckpt.model_checkpoint_path))
	saver.restore(sess, ckpt.model_checkpoint_path)

gs = sess.run(global_step)
sys.stdout = sys.__stdout__
print(sys.argv[1], end=": ")
print(gs, end=" | ")
to_test_op = to_testing(variable_averages)
sess.run(to_test_op)
meanError, stdError = testNetwork(sess, valError, batch_size, valIterator)

sys.stdout = sys.__stdout__
print(meanError, end=" - ")
print(stdError)
print(meanError, end=" - ", file=log)
print(stdError, file=log)

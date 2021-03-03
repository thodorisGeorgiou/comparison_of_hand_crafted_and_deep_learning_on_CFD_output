#Test singular network
import sys
import os
sys.stdout = open(os.devnull, "w")
sys.stderr = open(os.devnull, "w")
sys.stdwar = open(os.devnull, "w")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy
import tensorflow as tf
# sys.path.append("/tank/georgioutk/cliffordConvolutionMoreTest2/")
sys.path.append("/tank/georgioutk/cliffordConvolutionEddy/")
import cliffordConvolution as cc

import preprocessing
import modelsFullSkip as models
# import models

# import warnings
# tf.logging.set_verbosity(tf.logging.ERROR)

numGpus = 1
batch_size = 200
MOVING_AVERAGE_DECAY = 0.9999

# train_dir = os.getcwd()+"/"+sys.argv[1]
train_dir = sys.argv[1]
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
			count += testBatch_size
		except tf.errors.OutOfRangeError:
			break
	return numpy.sqrt(mean/count), std/count


global_step = tf.Variable(0, trainable=False)
currLr = 1
lr = tf.Variable(currLr, dtype=tf.float32, trainable=False)
tf.add_to_collection("learning_rate", lr)
# with warnings.catch_warnings():
	# warnings.simplefilter("ignore")
log = open(train_dir+"Accuracy4GPUHBS_test.log", "w", 1)
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

valPredictions = tf.concat(netOut, axis=0)
diff = tf.subtract(valPredictions, valData[-1])
valError = tf.reduce_sum(tf.square(diff), axis=0)
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

# print("Model Variables", file=log)
# for v in model_vars:
# 	print(v.name, file=log)

# print("Globa; Variables", file=log)
# for v in global_variables:
# 	print(v.name, file=log)

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

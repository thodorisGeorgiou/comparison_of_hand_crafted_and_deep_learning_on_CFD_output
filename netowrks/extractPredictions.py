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

numGpus = 1
batch_size = 200
MOVING_AVERAGE_DECAY = 0.9999

# train_dir = os.getcwd()+"/"+sys.argv[1]
# train_dir = sys.argv[1]
train_dir = "/tank/airfoil/experiments/netowrkswithValSet/2x512_3x128_2x64SkipPytorchBNormEpsilonvc/2Release"
modelType = "vc"
log = open("gettingPredictions.txt", "w", 1)
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

allOutputs = tf.concat(netOut, axis=0)
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
numpyPreds = []
numpyGt = []
print("Running")
while True:
	try:
		res = sess.run([allOutputs, valData[-1]])
		numpyPreds.append(res[0])
		numpyGt.append(res[1])
	except tf.errors.OutOfRangeError:
		break


numpyPreds = numpy.concatenate(numpyPreds, axis=0)
numpyGt = numpy.concatenate(numpyGt, axis=0)
numpy.sqrt(numpy.mean(numpy.square(numpyPreds-numpyGt), axis=0))
numpy.save("vcPredictions2", numpyPreds)
numpy.save("vcGt2", numpyGt)

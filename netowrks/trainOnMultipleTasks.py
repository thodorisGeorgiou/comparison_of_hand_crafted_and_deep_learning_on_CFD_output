#Train singular network on multiple tasks
import sys
import os
monitorOutput = sys.stdout
rubishOutput = open(os.devnull, "w")
sys.stdout = rubishOutput
sys.stderr = rubishOutput
sys.stdwar = rubishOutput
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy
import time
import tensorflow as tf

sys.path.append("/tank/georgioutk/cliffordConvolution/")
# sys.path.append("/tank/georgioutk/cliffordConvolutionMoreTest2/")
import cliffordConvolution as cc
import preprocessing
import modelsFullSkip as models
# import models

numGpus = 1
batch_size = 100
MOVING_AVERAGE_DECAY = 0.999
INITIAL_LEARNING_RATE = 1e-3

modelType = sys.argv[1]
run = sys.argv[2]

if modelType not in ["vc", "ds", "op", "cc"]:
	exit("Model type not supported")

train_dir = os.getcwd() + "TrainOnForceFlow/2x512_3x128_2x64SkipPytorchBNormEpsilon"+modelType+"/"+run

def ema_to_weights(ema, variables):
	return tf.group(*(tf.assign(var, ema.average(var).read_value()) for var in variables))

def save_weight_backups():
	return tf.group(*(tf.assign(bck, var.read_value()) for var, bck in zip(model_vars, backup_vars)))

def restore_weight_backups():
	return tf.group(*(tf.assign(var, bck.read_value()) for var, bck in zip(model_vars, backup_vars)))

def to_training():
	with tf.control_dependencies([tf.assign(is_training, True)]):
		return restore_weight_backups()

def to_testing(ema):
	with tf.control_dependencies([tf.assign(is_training, False)]):
		with tf.control_dependencies([save_weight_backups()]):
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
	if tf.gfile.Exists(train_dir):
		tf.gfile.DeleteRecursively(train_dir)
	tf.gfile.MakeDirs(train_dir)

	log = open(train_dir+".txt", "w", 1)
	is_training = tf.get_variable('is_training', shape=(), dtype=tf.bool, initializer=tf.constant_initializer(True, dtype=tf.bool), trainable=False)
	global_step = tf.Variable(0, trainable=False)

	trainDataSmall, testDataSmall, numTrainExamplesSmall, numTestExamplesSmall, testIteratorSmall = preprocessing.inputFlowsForFlowPrediction(batch_size)
	trainDataBig, testDataBig, numTrainExamplesBig, numTestExamplesBig, testIteratorBig = preprocessing.inputFlows(batch_size)
	# trainData, testData, numTrainExamples, numTestExamples, testIterator = preprocessing.inputFlows(batch_size, numTrainExamples=8000)
	# numTrainExamples = numTrainExamplesSmall
	numTrainExamples = numTrainExamplesBig

	perGPUTrainDataBig = [list([]) for i in range(numGpus)]
	for tD in trainDataBig:
		split = tf.split(tD, numGpus, axis=0)
		for gpu in range(numGpus):
			perGPUTrainDataBig[gpu].append(split[gpu])

	perGPUTestDataBig = [list([]) for i in range(numGpus)]
	for tD in testDataBig[:-1]:
		split = tf.split(tD, numGpus, axis=0)
		for gpu in range(numGpus):
			perGPUTestDataBig[gpu].append(split[gpu])


	perGPUTrainDataSmall = [list([]) for i in range(numGpus)]
	perGPUTrainLabelsSmall = [list([]) for i in range(numGpus)]
	for tD in trainDataSmall:
		gpuSplits = tf.split(tD, numGpus, axis=0)
		for gpu, gpuSplit in enumerate(gpuSplits):
			split = tf.split(gpuSplit, 2, axis=1)
			perGPUTrainDataSmall[gpu].append(split[0])
			perGPUTrainLabelsSmall[gpu].append(split[1])


	perGPUTestDataSmall = [list([]) for i in range(numGpus)]
	perGPUTestLabelsSmall = [list([]) for i in range(numGpus)]
	for tD in testDataSmall:
		gpuSplits = tf.split(tD, numGpus, axis=0)
		for gpu, gpuSplit in enumerate(gpuSplits):
			split = tf.split(gpuSplit, 2, axis=1)
			perGPUTestDataSmall[gpu].append(split[0])
			perGPUTestLabelsSmall[gpu].append(split[1])

	for gpu in range(numGpus):
		perGPUTrainLabelsSmall[gpu] = tf.concat(perGPUTrainLabelsSmall[gpu], axis=-1)
		perGPUTestLabelsSmall[gpu] = tf.concat(perGPUTestLabelsSmall[gpu], axis=-1)

	testLabelsSmall = tf.concat(perGPUTestLabelsSmall, axis=0)
	for gpu in range(numGpus):
		with tf.name_scope('tower_%d' % (gpu)) as scope:
			with tf.device('/gpu:%d' % gpu):
				print("Defining tower "+str(gpu))
				trainCodeBig = models.inference(perGPUTrainDataBig[gpu], first=(gpu==0), useType="train", modelType=modelType)
				trainCodeSmall = models.inference(perGPUTrainDataSmall[gpu], first=False, useType="train", modelType=modelType)
				flowPredictions = models.predictFlow(trainCodeSmall, batch_size//numGpus, log, useType="train", first=(gpu==0))
				forcePredictions = models.predictForces(trainCodeBig, batch_size//numGpus, log, useType="train", first=(gpu==0))
				# flowReconBig = models.reconstruct(trainCodeBig, batch_size//numGpus, log, useType="train", first=(gpu==0))
				# flowReconSmall = models.reconstruct(trainCodeSmall, batch_size//numGpus, log, useType="train", first=(gpu==0))
				# l2_loss = tf.nn.l2_loss(predictions - perGPUTrainData[gpu][-1], name="l2_loss_gpu_"+str(gpu))
				# print(perGPUTrainLabels[gpu].get_shape())
				weights = numpy.array([[10,1]])
				l2_loss_forces = tf.reduce_sum(tf.squared_difference(forcePredictions*weights, perGPUTrainDataBig[gpu][-1]*weights), name="l2_force_loss_gpu_"+str(gpu))
				l2_loss_flow = tf.reduce_sum(tf.squared_difference(flowPredictions, perGPUTrainLabelsSmall[gpu]), name="l2_flow_loss_gpu_"+str(gpu))
				# l2_flow_recon_loss = tf.reduce_sum(tf.squared_difference(flowReconSmall, tf.concat(perGPUTrainDataSmall[gpu], axis=-1)), name="l2_flow_recon_loss_gpu_"+str(gpu))
				# l2_force_recon_loss = tf.reduce_sum(tf.squared_difference(flowReconBig, tf.concat(perGPUTrainDataBig[gpu][:-1], axis=-1)), name="l2_force_recon_loss_gpu_"+str(gpu))
				tf.add_to_collection('l2_losses', l2_loss_forces)
				tf.add_to_collection('l2_losses', l2_loss_flow)
				# tf.add_to_collection('l2_losses', l2_flow_recon_loss)
				# tf.add_to_collection('l2_losses', l2_force_recon_loss)

	total_l2_loss = tf.reduce_mean(tf.get_collection('l2_losses'))
	tf.add_to_collection('losses', total_l2_loss)
	print("All towers defined.")

	netOutSmall = []
	netOutBig = []
	for gpu in range(numGpus):
		with tf.name_scope('tower_%d' % (gpu)) as scope:
			with tf.device('/gpu:%d' % gpu):
				testCodeSmall = models.inference(perGPUTestDataSmall[gpu], first=False, useType="test", modelType=modelType)
				testCodeBig = models.inference(perGPUTestDataBig[gpu], first=False, useType="test", modelType=modelType)
				gpuTestPredictionsSmall = models.predictFlow(testCodeSmall, batch_size//numGpus, log, useType="test", first=False)
				gpuTestPredictionsBig = models.predictForces(testCodeBig, batch_size//numGpus, log, useType="test", first=False)
				netOutSmall.append(gpuTestPredictionsSmall)
				netOutBig.append(gpuTestPredictionsBig)

	gpuTestPredictionsSmall = tf.concat(netOutSmall, axis=0)
	gpuTestPredictionsBig = tf.concat(netOutBig, axis=0)
	testErrorForce = tf.reduce_mean(tf.abs(tf.subtract(gpuTestPredictionsBig, testDataBig[-1])), axis=0)
	testErrorFlow = tf.reduce_mean(tf.square(tf.subtract(gpuTestPredictionsSmall, testLabelsSmall)))
	print("Test towers defined.")

	total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

	loss_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, name='avg')
	losses = tf.get_collection('losses')
	loss_averages_op = loss_averages.apply(losses + [total_loss])

	# Compute gradients.
	currLr = INITIAL_LEARNING_RATE
	lr = tf.Variable(INITIAL_LEARNING_RATE, dtype=tf.float32, trainable=False)
	with tf.control_dependencies([loss_averages_op]):
		opt = tf.train.AdamOptimizer(lr)
		# opt = tf.train.MomentumOptimizer(lr, 0.9)
		grads = opt.compute_gradients(total_loss, var_list=tf.trainable_variables(), colocate_gradients_with_ops=True)

	# Apply gradients.
	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

	# Track the moving averages of all trainable variables.
	model_vars = tf.trainable_variables()
	variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	variables_averages_op = variable_averages.apply(model_vars)

	for l in losses + [total_loss]:
		tf.summary.scalar(l.op.name +' (raw)', l)

	for l in tf.get_collection("l2_losses"):
		tf.summary.scalar(l.op.name +' (raw)', l)

	with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
		train_op = tf.no_op(name='train')

	with tf.variable_scope('BackupVariables'):
		backup_vars = [tf.get_variable(var.op.name, dtype=var.value().dtype, trainable=False, initializer=var.initialized_value()) for var in model_vars]

	regOps = tf.get_collection("regularizationOps")
	to_test_op = to_testing(variable_averages)
	to_train_op = to_training()

	saver = tf.train.Saver(tf.global_variables())
	saverMax = tf.train.Saver(tf.global_variables())

	init = tf.global_variables_initializer()
	myconfig = tf.ConfigProto(log_device_placement=False)
	myconfig.gpu_options.allow_growth = True
	sess = tf.Session(config=myconfig)

	writer = tf.summary.FileWriter(train_dir, sess.graph)
	writerMax = tf.summary.FileWriter(train_dir+"Release/", sess.graph)
	sess.run(init)
	_summ = tf.summary.merge_all()

	min_drag = numpy.finfo(numpy.float32).max
	meanDragError = None
	min_lift = numpy.finfo(numpy.float32).max
	meanLiftError = None
	min_flow = numpy.finfo(numpy.float32).max
	meanFlowError = None
	SuccRate_summary = tf.Summary()
	SuccRate_summary.value.add(tag='flow_error', simple_value=meanFlowError)
	SuccRate_summary.value.add(tag='drag_error', simple_value=meanDragError)
	SuccRate_summary.value.add(tag='lift_error', simple_value=meanLiftError)
	SuccRate_summary.value.add(tag='min_flow_error', simple_value=min_flow)
	SuccRate_summary.value.add(tag='min_drag_error', simple_value=min_drag)
	SuccRate_summary.value.add(tag='min_lift_error', simple_value=min_lift)

	totalSteps = int(2*240*numTrainExamples/batch_size)
	for step in range(totalSteps):
		# if step==8000:
		# 	currLr = 5e-4
		# 	print("learning rate = "+str(currLr), file=log)
		# 	lr.load(currLr, sess)
		# if step==3000:
		# 	currLr /= 10
		# 	print("learning rate = "+str(currLr), file=log)
		# 	lr.load(currLr, sess)
		# if step==6000:
		# 	currLr /= 10
		# 	print("learning rate = "+str(currLr), file=log)
		# 	lr.load(currLr, sess)
		# if step==9000:
		# 	currLr /= 10
		# 	print("learning rate = "+str(currLr), file=log)
		# 	lr.load(currLr, sess)
		__ = sess.run(regOps)
		l2Loss, totalLoss, summ, _ = sess.run([total_l2_loss, total_loss, _summ, train_op])
		writer.add_summary(summ, step)
		assert not numpy.any(numpy.isnan(totalLoss)), "NaN Loss"
		print(str(step)+" "+str(l2Loss), file=log)
		if step % (numTrainExamples//(batch_size*0.25)) == 0:
			sys.stdout = monitorOutput
			print("%2.2f"%(step*100/totalSteps), end="\r", flush=True)
			sys.stdout = rubishOutput
			checkpoint_path = os.path.join(train_dir, 'model.ckpt')
			saver.save(sess, checkpoint_path, global_step=step)
		if step % (numTrainExamples//(batch_size*0.25)) == 0 and step != 0:
			sess.run(to_test_op)
			meanError, stdError = testNetwork(sess, testErrorForce, batch_size, testIteratorBig)
			meanFlowError, stdFlowError = testNetwork(sess, testErrorFlow, batch_size, testIteratorSmall)
			meanDragError = meanError[0]
			meanLiftError = meanError[1]
			print("Test forces:"+str(meanError)+" +- "+str(stdError)+"/"+str(min_drag)+" - "+str(min_lift), file=log)
			print("Test flow:"+str(meanFlowError)+" +- "+str(stdFlowError)+"/"+str(min_flow), file=log)
			# if (meanFlowError) < (min_flow):
			if (meanDragError + meanLiftError + meanFlowError) < (min_drag + min_lift + min_flow):
				min_flow = meanFlowError
				min_drag = meanDragError
				min_lift = meanLiftError
				checkpoint_path = os.path.join(train_dir+"Release/", 'model.ckpt')
				saverMax.save(sess, checkpoint_path, global_step=step)
				writerMax.add_summary(summ, step)
			SuccRate_summary.value[0].simple_value = meanFlowError
			SuccRate_summary.value[1].simple_value = meanDragError
			SuccRate_summary.value[2].simple_value = meanLiftError
			SuccRate_summary.value[3].simple_value = min_flow
			SuccRate_summary.value[4].simple_value = min_drag
			SuccRate_summary.value[5].simple_value = min_lift
			writer.add_summary(SuccRate_summary, step)
			sess.run(to_train_op)

	print("Saving..")
	checkpoint_path = os.path.join(train_dir, 'model.ckpt')
	saver.save(sess, checkpoint_path, global_step=step)
	sess.run(to_test_op)
	meanError, stdError = testNetwork(sess, testErrorForce, batch_size, testIteratorBig)
	meanFlowError, stdFlowError = testNetwork(sess, testErrorFlow, batch_size, testIteratorSmall)
	meanDragError = meanError[0]
	meanLiftError = meanError[1]
	print("Test forces:"+str(meanError)+" +- "+str(stdError)+"/"+str(min_drag)+" - "+str(min_lift), file=log)
	print("Test flow:"+str(meanFlowError)+" +- "+str(stdFlowError)+"/"+str(min_flow), file=log)
	# if (meanFlowError) < (min_flow):
	if (meanDragError + meanLiftError + meanFlowError) < (min_drag + min_lift + min_flow):
		min_drag = meanDragError
		min_lift = meanLiftError
		min_flow = meanFlowError
		checkpoint_path = os.path.join(train_dir+"Release/", 'model.ckpt')
		saverMax.save(sess, checkpoint_path, global_step=step)
		writerMax.add_summary(summ, step)
	SuccRate_summary.value[0].simple_value = meanFlowError
	SuccRate_summary.value[1].simple_value = meanDragError
	SuccRate_summary.value[2].simple_value = meanLiftError
	SuccRate_summary.value[3].simple_value = min_flow
	SuccRate_summary.value[4].simple_value = min_drag
	SuccRate_summary.value[5].simple_value = min_lift
	writer.add_summary(SuccRate_summary, step)

	time.sleep(10)

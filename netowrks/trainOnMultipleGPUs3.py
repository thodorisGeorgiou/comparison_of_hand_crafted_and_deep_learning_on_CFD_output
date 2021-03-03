#Train single network on multiple GPUs
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

numGpus = 1
batch_size = 100
MOVING_AVERAGE_DECAY = 0.999
INITIAL_LEARNING_RATE = 1e-3

modelType = sys.argv[1]
run = sys.argv[2]

if modelType not in ["vc", "ds", "op", "cc"]:
	exit("Model type not supported")

train_dir = os.getcwd() + "adam/512_3x128_2x64SkipPytorchBNormEpsilon"+modelType+"/"+run

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

	trainData, testData, numTrainExamples, numTestExamples, testIterator = preprocessing.inputFlows(batch_size)

	perGPUTrainData = [list([]) for i in range(numGpus)]
	for tD in trainData:
		split = tf.split(tD, numGpus, axis=0)
		for gpu in range(numGpus):
			perGPUTrainData[gpu].append(split[gpu])

	perGPUTestData = [list([]) for i in range(numGpus)]
	for tD in testData[:-1]:
		split = tf.split(tD, numGpus, axis=0)
		for gpu in range(numGpus):
			perGPUTestData[gpu].append(split[gpu])

	for gpu in range(numGpus):
		with tf.name_scope('tower_%d' % (gpu)) as scope:
			with tf.device('/gpu:%d' % gpu):
				print("Defining tower "+str(gpu))
				print(perGPUTrainData[gpu][0].get_shape())
				print(len(perGPUTrainData[gpu]))
				trainCode = models.inference(perGPUTrainData[gpu][:-1], first=(gpu==0), useType="train", modelType=modelType)
				print(trainCode.get_shape())
				predictions = models.predictForces(trainCode, batch_size//numGpus, log, useType="train", first=(gpu==0))
				# l2_loss = tf.nn.l2_loss(predictions - perGPUTrainData[gpu][-1], name="l2_loss_gpu_"+str(gpu))
				weights = numpy.array([[10,1]])
				l2_loss = tf.reduce_sum(tf.squared_difference(predictions*weights, perGPUTrainData[gpu][-1]*weights), name="l2_loss_gpu_"+str(gpu))
				tf.add_to_collection('l2_losses', l2_loss)

	total_l2_loss = tf.reduce_mean(tf.get_collection('l2_losses'))
	tf.add_to_collection('losses', total_l2_loss)
	print("All towers defined.")

	netOut = []
	for gpu in range(numGpus):
		with tf.name_scope('tower_%d' % (gpu)) as scope:
			with tf.device('/gpu:%d' % gpu):
				print(perGPUTestData[gpu][0].get_shape())
				print(len(perGPUTestData[gpu]))
				testCode = models.inference(perGPUTestData[gpu], first=False, useType="test", modelType=modelType)
				print(testCode.get_shape())
				gpuTestPredictions = models.predictForces(testCode, batch_size//numGpus, log, useType="test", first=False)
				netOut.append(gpuTestPredictions)

	testPredictions = tf.concat(netOut, axis=0)
	# testError = tf.reduce_mean(tf.abs(tf.subtract(testPredictions, testData[-1])), axis=0)
	testError = tf.reduce_mean(tf.square(tf.subtract(testPredictions, testData[-1])), axis=0)
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

	# regOps = tf.get_collection("regularizationOps")
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
	SuccRate_summary = tf.Summary()
	SuccRate_summary.value.add(tag='drag_error', simple_value=meanDragError)
	SuccRate_summary.value.add(tag='lift_error', simple_value=meanLiftError)
	SuccRate_summary.value.add(tag='min_drag_error', simple_value=min_drag)
	SuccRate_summary.value.add(tag='min_lift_error', simple_value=min_lift)

	totalSteps = int(240*numTrainExamples/batch_size)
	for step in range(totalSteps):
		# if step==300:
		# 	currLr = 1e-2
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
		# __ = sess.run(regOps)
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
			meanError, stdError = testNetwork(sess, testError, batch_size, testIterator)
			meanDragError = meanError[0]
			meanLiftError = meanError[1]
			print("Test :"+str(meanError)+" +- "+str(stdError)+"/"+str(min_drag)+" - "+str(min_lift), file=log)
			if (meanDragError + meanLiftError) < (min_drag + min_lift):
				min_drag = meanDragError
				min_lift = meanLiftError
				checkpoint_path = os.path.join(train_dir+"Release/", 'model.ckpt')
				saverMax.save(sess, checkpoint_path, global_step=step)
				writerMax.add_summary(summ, step)
			SuccRate_summary.value[0].simple_value = meanDragError
			SuccRate_summary.value[1].simple_value = meanLiftError
			SuccRate_summary.value[2].simple_value = min_drag
			SuccRate_summary.value[3].simple_value = min_lift
			writer.add_summary(SuccRate_summary, step)
			sess.run(to_train_op)

	print("Saving..")
	checkpoint_path = os.path.join(train_dir, 'model.ckpt')
	saver.save(sess, checkpoint_path, global_step=step)
	sess.run(to_test_op)
	meanError, stdError = testNetwork(sess, testError, batch_size, testIterator)
	meanDragError = meanError[0]
	meanLiftError = meanError[1]
	print("Test :"+str(meanError)+" +- "+str(stdError)+"/"+str(min_drag)+" - "+str(min_lift), file=log)
	if (meanDragError + meanLiftError) < (min_drag + min_lift):
		min_drag = meanDragError
		min_lift = meanLiftError
		checkpoint_path = os.path.join(train_dir+"Release/", 'model.ckpt')
		saverMax.save(sess, checkpoint_path, global_step=step)
		writerMax.add_summary(summ, step)
	SuccRate_summary.value[0].simple_value = meanDragError
	SuccRate_summary.value[1].simple_value = meanLiftError
	SuccRate_summary.value[2].simple_value = min_drag
	SuccRate_summary.value[3].simple_value = min_lift
	writer.add_summary(SuccRate_summary, step)

	time.sleep(10)

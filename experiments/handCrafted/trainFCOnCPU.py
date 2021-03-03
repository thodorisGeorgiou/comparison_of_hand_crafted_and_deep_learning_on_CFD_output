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

run = sys.argv[1]
train_dir = os.getcwd() + "/" + descriptor + "mlp/"+run

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

	# try:
	# 	testSet = numpy.load("testSetInUint8.npy")
	# 	testLabels = numpy.load(preprocessing.testLabelPath).astype(numpy.float32)
	# except FileNotFoundError:
	# 	testSet, testLabels = preprocessing.loadTestSet()
	# 	numpy.save("testSetInUint8.npy", testSet)

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

	if tf.gfile.Exists(train_dir):
		tf.gfile.DeleteRecursively(train_dir)
	tf.gfile.MakeDirs(train_dir)

	log = open(train_dir+".txt", "w", 1)
	is_training = tf.get_variable('is_training', shape=(), dtype=tf.bool, initializer=tf.constant_initializer(True, dtype=tf.bool), trainable=False)
	global_step = tf.Variable(0, trainable=False)

	dataset = tf.data.Dataset.from_tensor_slices((tfIdfTrain.astype(numpy.float32), trainLabels))
	dataset = dataset.shuffle(buffer_size=15000)
	dataset = dataset.repeat()
	dataset = dataset.batch(batch_size)
	dataset = dataset.prefetch(10)
	iterator = dataset.make_one_shot_iterator()

	trainCode, batchTrainLabels = iterator.get_next()
	predictions = models.predictForces(trainCode, batch_size, log, useType="train", first=True)
	weights = numpy.array([[10,1]])
	l2_loss = tf.reduce_sum(tf.squared_difference(predictions*weights, batchTrainLabels*weights), name="l2_loss")
	tf.add_to_collection('losses', l2_loss)

	valDataset = tf.data.Dataset.from_tensor_slices((tfIdfVal.astype(numpy.float32), valLabels))
	valDataset = valDataset.batch(batch_size)
	valIterator = valDataset.make_initializable_iterator()
	valCode, batchValLabels = valIterator.get_next()
	valPredictions = models.predictForces(valCode, batch_size, log, useType="test", first=False)

	testError = tf.reduce_mean(tf.square(tf.subtract(valPredictions, batchValLabels)), axis=0)
	print("Towers defined.")
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

	totalSteps = int(240*trainSetSize/batch_size)
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
		l2Loss, totalLoss, summ, _ = sess.run([l2_loss, total_loss, _summ, train_op])
		writer.add_summary(summ, step)
		assert not numpy.any(numpy.isnan(totalLoss)), "NaN Loss"
		print(str(step)+" "+str(l2Loss), file=log)
		if step % (trainSetSize//(batch_size*0.25)) == 0:
			print("%2.2f"%(step*100/totalSteps), end="\r", flush=True)
			checkpoint_path = os.path.join(train_dir, 'model.ckpt')
			saver.save(sess, checkpoint_path, global_step=step)
		if step % (trainSetSize//(batch_size*0.25)) == 0 and step != 0:
			sess.run(to_test_op)
			meanError, stdError = testNetwork(sess, testError, batch_size, valIterator)
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
	meanError, stdError = testNetwork(sess, testError, batch_size, valIterator)
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

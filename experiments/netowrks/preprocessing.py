#Data preprocessing script
import os
import numpy
import tensorflow as tf
# import cliffordConvolution as cc

trainSetPath = "/tank/airfoil/sets/trainSet"
testSetPath = "/tank/airfoil/sets/testSet"
valSetPath = "/tank/airfoil/sets/valSet"
trainLabelPath = "/tank/airfoil/sets/trainGroundTruth.npy"
testLabelPath = "/tank/airfoil/sets/testGroundTruth.npy"
valLabelPath = "/tank/airfoil/sets/valGroundTruth.npy"

def normalizeTensor(tensor, width, height):
	tensorShape = tensor.get_shape()
	norm = tf.norm(tensor)
	norm = tf.where(tf.equal(norm, 0), tf.ones_like(norm), norm)
	tensor = tensor*tf.sqrt(tensorShape[0].value*tensorShape[1].value/(width*height))/norm
	return tensor

def preprocerssFlowForTraining(flowPath, label):
	reader = tf.read_file(flowPath)
	data = tf.decode_raw(reader, tf.float32)
	res = [128+64, 128, 6]
	data = tf.reshape(data, res)
	data = tf.random_crop(data, [160, 120, 6])
	ux, uy, uz, p, nut, nuTilda = tf.unstack(data, axis=-1)
	u = tf.stack([ux,uz], axis=-1)
	u = normalizeTensor(u, 5, 5)
	p = tf.expand_dims(normalizeTensor(p, 5, 5), axis=-1)
	nut = tf.expand_dims(normalizeTensor(nut, 5, 5), axis=-1)
	nuTilda = tf.expand_dims(normalizeTensor(nuTilda, 5, 5), axis=-1)
	return u, p, nut, nuTilda, label

def preprocerssFlowForTrainingOnFlow(flowPath):
	reader = tf.read_file(flowPath)
	data = tf.decode_raw(reader, tf.float32)
	res = [128+64, 128, 6]
	data = tf.reshape(data, res)
	data = tf.random_crop(data, [96, 64, 6])
	ux, uy, uz, p, nut, nuTilda = tf.unstack(data, axis=-1)
	u = tf.stack([ux,uz], axis=-1)
	u = normalizeTensor(u, 5, 5)
	p = tf.expand_dims(normalizeTensor(p, 5, 5), axis=-1)
	nut = tf.expand_dims(normalizeTensor(nut, 5, 5), axis=-1)
	nuTilda = tf.expand_dims(normalizeTensor(nuTilda, 5, 5), axis=-1)
	return u, p, nut, nuTilda

def preprocerssFlowForTest(flowPath, label):
	reader = tf.read_file(flowPath)
	data = tf.decode_raw(reader, tf.float32)
	res = [128+64, 128, 6]
	data = tf.reshape(data, res)
	data = tf.image.resize_image_with_crop_or_pad(data, 160, 120)
	ux, uy, uz, p, nut, nuTilda = tf.unstack(data, axis=-1)
	u = tf.stack([ux,uz], axis=-1)
	u = normalizeTensor(u, 5, 5)
	p = tf.expand_dims(normalizeTensor(p, 5, 5), axis=-1)
	nut = tf.expand_dims(normalizeTensor(nut, 5, 5), axis=-1)
	nuTilda = tf.expand_dims(normalizeTensor(nuTilda, 5, 5), axis=-1)
	return u, p, nut, nuTilda, label

def preprocerssFlowForTestOnFlow(flowPath):
	reader = tf.read_file(flowPath)
	data = tf.decode_raw(reader, tf.float32)
	res = [128+64, 128, 6]
	data = tf.reshape(data, res)
	crops = [tf.image.resize_image_with_crop_or_pad(data, 96, 64)]
	inds = [[0,0,0], [res[0]-96,0,0],[0,res[1]-64,0],[res[0]-96,res[1]-64,0]]
	for startPoint in inds:
		crops.append(tf.slice(data, startPoint, [96,64,6]))
	u = []
	p = []
	nut = []
	nuTilda = []
	for crop in crops:
		ux, uy, uz, pc, nutc, nuTildac = tf.unstack(crop, axis=-1)
		up = tf.stack([ux,uz], axis=-1)
		u.append(normalizeTensor(up, 5, 5))
		p.append(tf.expand_dims(normalizeTensor(pc, 5, 5), axis=-1))
		nut.append(tf.expand_dims(normalizeTensor(nutc, 5, 5), axis=-1))
		nuTilda.append(tf.expand_dims(normalizeTensor(nuTildac, 5, 5), axis=-1))
	return tf.stack(u, axis=0), tf.stack(p, axis=0), tf.stack(nut, axis=0), tf.stack(nuTilda, axis=0)

def preprocerssFlowForTestOnFlowAllCrops(flowPath):
	reader = tf.read_file(flowPath)
	data = tf.decode_raw(reader, tf.float32)
	res = [128+64, 128, 6]
	data = tf.reshape(data, res)
	# crops = [tf.image.resize_image_with_crop_or_pad(data, 96, 64)]
	crops = []
	inds = [[i,j,0] for i in range(0,res[0]-96,4) for j in range(0,res[1]-64,4)]
	for startPoint in inds:
		crops.append(tf.slice(data, startPoint, [96,64,6]))
	u = []
	p = []
	nut = []
	nuTilda = []
	for crop in crops:
		ux, uy, uz, pc, nutc, nuTildac = tf.unstack(crop, axis=-1)
		up = tf.stack([ux,uz], axis=-1)
		u.append(normalizeTensor(up, 5, 5))
		p.append(tf.expand_dims(normalizeTensor(pc, 5, 5), axis=-1))
		nut.append(tf.expand_dims(normalizeTensor(nutc, 5, 5), axis=-1))
		nuTilda.append(tf.expand_dims(normalizeTensor(nuTildac, 5, 5), axis=-1))
	return tf.stack(u, axis=0), tf.stack(p, axis=0), tf.stack(nut, axis=0), tf.stack(nuTilda, axis=0)

def preprocersSingleFlowForTestMutiCrop(data):
	ux, uy, uz, p, nut, nuTilda = tf.unstack(data, axis=-1)
	u = tf.stack([ux,uz], axis=-1)
	u = normalizeTensor(u, 5, 5)
	p = tf.expand_dims(normalizeTensor(p, 5, 5), axis=-1)
	nut = tf.expand_dims(normalizeTensor(nut, 5, 5), axis=-1)
	nuTilda = tf.expand_dims(normalizeTensor(nuTilda, 5, 5), axis=-1)
	return u, p, nut, nuTilda

def preprocerssFlowForTestMutiCrop(flowPath, label):
	reader = tf.read_file(flowPath)
	data = tf.decode_raw(reader, tf.float32)
	res = [128+64, 128, 6]
	data = tf.reshape(data, res)
	allCrops = []
	allCrops.append(preprocersSingleFlowForTestMutiCrop(tf.image.resize_image_with_crop_or_pad(data, 160, 120)))
	allCrops.append(preprocersSingleFlowForTestMutiCrop(tf.image.crop_to_bounding_box(data, 0, 0, 160, 120)))
	allCrops.append(preprocersSingleFlowForTestMutiCrop(tf.image.crop_to_bounding_box(data, 0, 128-120, 160, 120)))
	allCrops.append(preprocersSingleFlowForTestMutiCrop(tf.image.crop_to_bounding_box(data, 192-160, 0, 160, 120)))
	allCrops.append(preprocersSingleFlowForTestMutiCrop(tf.image.crop_to_bounding_box(data, 192-160, 128-120, 160, 120)))
	u = tf.stack([f[0] for f in allCrops], axis=0)
	p = tf.stack([f[1] for f in allCrops], axis=0)
	nut = tf.stack([f[2] for f in allCrops], axis=0)
	nuTilda = tf.stack([f[3] for f in allCrops], axis=0)
	return u, p, nut, nuTilda, label

def inputFlows(batch_size, numTrainExamples=None):
	trainPaths = os.listdir(trainSetPath)
	trainIndeces = []
	for p in range(len(trainPaths)):
		trainIndeces.append(int(trainPaths[p].split(".")[0]))
		trainPaths[p] = trainSetPath+"/"+trainPaths[p]
	if numTrainExamples is not None:
		sortedTrainIndeces = numpy.argsort(trainIndeces)[:numTrainExamples]
	else:
		sortedTrainIndeces = numpy.argsort(trainIndeces)
	trainPaths = numpy.array(trainPaths)[sortedTrainIndeces]
	# testPaths = os.listdir(testSetPath)
	testPaths = os.listdir(valSetPath)
	testIndeces = []
	for p in range(len(testPaths)):
		testIndeces.append(int(testPaths[p].split(".")[0]))
		testPaths[p] = valSetPath+"/"+testPaths[p]
	sortedTestIndeces = numpy.argsort(testIndeces)
	testPaths = numpy.array(testPaths)[sortedTestIndeces]
	# testPaths = os.listdir(testSetPath)
	# for p in range(len(testPaths)):
	# 	testPaths[p] = testSetPath+"/"+testPaths[p]
	if numTrainExamples is not None:
		trainLabels = numpy.load(trainLabelPath).astype(numpy.float32)[:numTrainExamples]
	else:
		trainLabels = numpy.load(trainLabelPath).astype(numpy.float32)
	testLabels = numpy.load(valLabelPath).astype(numpy.float32)
	# testLabels = numpy.load(testLabelPath).astype(numpy.float32)
	numTrainExamples = len(trainPaths)
	numTestExamples = len(testPaths)

	dataset = tf.data.Dataset.from_tensor_slices((trainPaths, trainLabels))
	dataset = dataset.map(preprocerssFlowForTraining, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	dataset = dataset.shuffle(buffer_size=15000)
	dataset = dataset.repeat()
	dataset = dataset.batch(batch_size)
	dataset = dataset.prefetch(10)
	iterator = dataset.make_one_shot_iterator()
	# iterator = dataset.make_initializable_iterator()
	trainData = []
	for tD in iterator.get_next():
		trainData.append(tf.reshape(tD, [batch_size]+tD.get_shape().as_list()[1:]))

	testDataset = tf.data.Dataset.from_tensor_slices((testPaths, testLabels))
	testDataset = testDataset.map(preprocerssFlowForTest, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	testDataset = testDataset.batch(batch_size)
	testIterator = testDataset.make_initializable_iterator()
	testData = []
	for tD in testIterator.get_next():
		testData.append(tf.reshape(tD, [batch_size]+tD.get_shape().as_list()[1:]))

	return trainData, testData, numTrainExamples, numTestExamples, testIterator

def inputValFlows(batch_size, path, labelPath):
	valPaths = os.listdir(path)
	valIndeces = []
	for p in range(len(valPaths)):
		valIndeces.append(int(valPaths[p].split(".")[0]))
		valPaths[p] = path+"/"+valPaths[p]
	sortedValIndeces = numpy.argsort(valIndeces)
	valPaths = numpy.array(valPaths)[sortedValIndeces]

	valLabels = numpy.load(labelPath).astype(numpy.float32)
	numValExamples = len(valPaths)

	valDataset = tf.data.Dataset.from_tensor_slices((valPaths, valLabels))
	valDataset = valDataset.map(preprocerssFlowForTest, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	# valDataset = valDataset.map(preprocerssFlowForTestMutiCrop, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	valDataset = valDataset.batch(batch_size)
	valIterator = valDataset.make_initializable_iterator()
	valData = []
	for tD in valIterator.get_next():
		valData.append(tf.reshape(tD, [batch_size]+tD.get_shape().as_list()[1:]))
	# for tD in valIterator.get_next():
	# 	shape = tD.get_shape().as_list()
	# 	if len(shape) > 2:
	# 		valData.append(tf.reshape(tD, [batch_size*5]+shape[2:]))
	# 	else:
	# 		valData.append(tf.reshape(tD, [batch_size]+shape[1:]))

	return valData, numValExamples, valIterator

def inputValFlowsForFlowPrediction(batch_size, path):
	valPaths = os.listdir(path)
	valIndeces = []
	for p, pt in enumerate(valPaths):
		valIndeces.append(int(pt.split(".")[0]))
		valPaths[p] = path+"/"+pt
	sortedValIndeces = numpy.argsort(valIndeces)
	valPaths = numpy.array(valPaths)[sortedValIndeces]

	numValExamples = len(valPaths)

	valDataset = tf.data.Dataset.from_tensor_slices((valPaths))
	valDataset = valDataset.map(preprocerssFlowForTestOnFlowAllCrops, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	# valDataset = valDataset.map(preprocerssFlowForTestMutiCrop, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	valDataset = valDataset.batch(batch_size)
	valIterator = valDataset.make_initializable_iterator()
	valData = []
	for tD in valIterator.get_next():
		valData.append(tf.reshape(tD, [batch_size*384]+tD.get_shape().as_list()[2:]))

	# for tD in valIterator.get_next():
	# 	shape = tD.get_shape().as_list()
	# 	if len(shape) > 2:
	# 		valData.append(tf.reshape(tD, [batch_size*5]+shape[2:]))
	# 	else:
	# 		valData.append(tf.reshape(tD, [batch_size]+shape[1:]))

	return valData, numValExamples, valIterator

def inputFlowsForFlowPrediction(batch_size, numTrainExamples=None):
	trainPaths = os.listdir(trainSetPath)
	trainIndeces = []
	for p in range(len(trainPaths)):
		trainIndeces.append(int(trainPaths[p].split(".")[0]))
		trainPaths[p] = trainSetPath+"/"+trainPaths[p]
	if numTrainExamples is not None:
		sortedTrainIndeces = numpy.argsort(trainIndeces)[:numTrainExamples]
	else:
		sortedTrainIndeces = numpy.argsort(trainIndeces)
	trainPaths = numpy.array(trainPaths)[sortedTrainIndeces]
	# testPaths = os.listdir(testSetPath)
	testPaths = os.listdir(valSetPath)
	testIndeces = []
	for p in range(len(testPaths)):
		testIndeces.append(int(testPaths[p].split(".")[0]))
		testPaths[p] = valSetPath+"/"+testPaths[p]
	sortedTestIndeces = numpy.argsort(testIndeces)
	testPaths = numpy.array(testPaths)[sortedTestIndeces]
	# testPaths = os.listdir(testSetPath)
	# for p in range(len(testPaths)):
	# 	testPaths[p] = testSetPath+"/"+testPaths[p]
	if numTrainExamples is not None:
		trainLabels = numpy.load(trainLabelPath).astype(numpy.float32)[:numTrainExamples]
	else:
		trainLabels = numpy.load(trainLabelPath).astype(numpy.float32)
	testLabels = numpy.load(valLabelPath).astype(numpy.float32)
	# testLabels = numpy.load(testLabelPath).astype(numpy.float32)
	numTrainExamples = len(trainPaths)
	numTestExamples = len(testPaths)*5

	dataset = tf.data.Dataset.from_tensor_slices((trainPaths))
	dataset = dataset.map(preprocerssFlowForTrainingOnFlow, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	dataset = dataset.shuffle(buffer_size=15000)
	dataset = dataset.repeat()
	dataset = dataset.batch(batch_size)
	dataset = dataset.prefetch(10)
	iterator = dataset.make_one_shot_iterator()
	# iterator = dataset.make_initializable_iterator()
	trainData = []
	for tD in iterator.get_next():
		trainData.append(tf.reshape(tD, [batch_size]+tD.get_shape().as_list()[1:]))

	testDataset = tf.data.Dataset.from_tensor_slices((testPaths))
	testDataset = testDataset.map(preprocerssFlowForTestOnFlow, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	testDataset = testDataset.batch(batch_size)
	testIterator = testDataset.make_initializable_iterator()
	testData = []
	for tD in testIterator.get_next():
		testData.append(tf.reshape(tD, [batch_size*5]+tD.get_shape().as_list()[2:]))

	return trainData, testData, numTrainExamples, numTestExamples, testIterator

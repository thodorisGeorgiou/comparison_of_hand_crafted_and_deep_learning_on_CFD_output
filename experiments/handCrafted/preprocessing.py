import os
import numpy

#Set data paths
trainSetPath = "sets/trainSet"
testSetPath = "sets/testSet"
valSetPath = "sets/valSet"
trainLabelPath = "sets/trainGroundTruth.npy"
testLabelPath = "sets/testGroundTruth.npy"
valLabelPath = "sets/valGroundTruth.npy"

def loadData(flowPath, flowLabelPath):
	flowPaths = os.listdir(flowPath)
	flowIndeces = []
	for p, path in enumerate(flowPaths):
		flowIndeces.append(int(path.split(".")[0]))
		flowPaths[p] = flowPath+"/"+path
	sortedIndeces = numpy.argsort(flowIndeces)
	flowPaths = numpy.array(flowPaths)[sortedIndeces]
	res = [128+64, 128, 6]
	flowSet = []
	for path in flowPaths:
		f = numpy.fromfile(path, dtype=numpy.float32)
		f = numpy.reshape(f, res)
		f = numpy.split(f, 6, axis=-1)
		del(f[1])
		for i in range(5):
			f[i][:,:] -= numpy.min(f[i][:,:])
			f[i][:,:] /= numpy.max(f[i][:,:])
			f[i][:,:] = f[i][:,:]*255
		flowSet.append((numpy.concatenate(f, axis=-1)).astype(numpy.uint8))
	flowLabels = numpy.load(flowLabelPath).astype(numpy.float32)
	return numpy.array(flowSet), flowLabels

def loadTrainSet():
	return loadData(trainSetPath, trainLabelPath)

def loadTestSet():
	return loadData(testSetPath, testLabelPath)

def loadValSet():
	return loadData(valSetPath, valLabelPath)


#Load flow fields. OpenCV expects images in uint8. Flwo fields are comprised by floating points. First time we load the fields, transform them to uint8 and save them so we don't need to do it again.
#If the uint8 version exists load it.
def loadTrainAndTestSets():
	print("Loading Data")
	try:
		trainSet = numpy.load("trainSetInUint8.npy")
		trainLabels = numpy.load(trainLabelPath).astype(numpy.float32)
	except FileNotFoundError:
		trainSet, trainLabels = loadTrainSet()
		numpy.save("trainSetInUint8.npy", trainSet)
	try:
		testSet = numpy.load("testSetInUint8.npy")
		testLabels = numpy.load(testLabelPath).astype(numpy.float32)
	except FileNotFoundError:
		testSet, testLabels = loadTestSet()
		numpy.save("testSetInUint8.npy", testSet)
	return [trainSet, trainLabels], [testSet, testLabels]
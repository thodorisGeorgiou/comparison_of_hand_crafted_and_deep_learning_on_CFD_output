import os
import random
import numpy
import pickle
from matplotlib import pyplot as plt

# mainDirs = ["/home/prometheus/thodorisGeorgiou/airfoil/dataset/cases7", "/home/prometheus/thodorisGeorgiou/airfoil/dataset/cases8"]
mainDirs = ["/tank/airfoil/dataset/cases7", "/tank/airfoil/dataset/cases8"]
cases = [md+"/"+ld for md in mainDirs for ld in os.listdir(md)]
files = ["run_SA_Re1e6_coarse_Nx100_ExpT1500_ExpD300_a"+str(i) for i in [-8, -6, -4, -2, 2, 4, 6, 8]]

lift = []
drag = []

index = 0
trainGroundTruth = numpy.zeros([15000, 2])
testGroundTruth = numpy.zeros([1000, 2])
allExamples = [c+"/"+f for f in files for c in cases]
random.shuffle(allExamples)
trainSetToCasesMap = {}
testSetToCasesMap = {}
for example in allExamples:
	print(str(index), end="\r", flush=True)
	forces = numpy.loadtxt(example+"/postProcessing/forces/0/forceCoeffs.dat")
	# flow = numpy.fromfile(example+"/postProcessing/sampleDict/8000/allInOne.raw", dtype=numpy.float32)
	# flow = numpy.reshape(flow, [192, 128, 6])
	if index < 15000:
		os.system("cp "+example+"/postProcessing/sampleDict/8000/allInOne.raw"+" trainSet/"+str(index)+".raw")
		# numpy.save("trainSet/"+str(index)+".npy", flow)
		trainGroundTruth[index] = forces[-1][2:4]
		trainSetToCasesMap[index] = example
	else:
		testIndex = index - 15000
		os.system("cp "+example+"/postProcessing/sampleDict/8000/allInOne.raw"+" testSet/"+str(testIndex)+".raw")
		# numpy.save("testSet/"+str(testIndex)+".npy", flow)
		testGroundTruth[testIndex] = forces[-1][2:4]
		testSetToCasesMap[testIndex] = example
	index += 1

pickle.dump(trainSetToCasesMap, open("trainSetToCasesMap.pkl", "wb"))
numpy.save("trainGroundTruth.npy", trainGroundTruth)

pickle.dump(testSetToCasesMap, open("testSetToCasesMap.pkl", "wb"))
numpy.save("testGroundTruth.npy", testGroundTruth)

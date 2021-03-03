import os
import random
import numpy
import pickle
from matplotlib import pyplot as plt

# mainDirs = ["/home/prometheus/thodorisGeorgiou/airfoil/dataset/cases7", "/home/prometheus/thodorisGeorgiou/airfoil/dataset/cases8"]
mainDirs = ["/tank/airfoil/dataset/cases9"]
cases = [md+"/"+ld for md in mainDirs for ld in os.listdir(md)]
files = ["run_SA_Re1e6_coarse_Nx100_ExpT1500_ExpD300_a"+str(i) for i in [-8, -6, -4, -2, 2, 4, 6, 8]]

lift = []
drag = []

index = 0
groundTruth = numpy.zeros([1000, 2])
allExamples = [c+"/"+f for f in files for c in cases]
random.shuffle(allExamples)
valSetToCasesMap = {}
for example in allExamples:
	print(str(index), end="\r", flush=True)
	forces = numpy.loadtxt(example+"/postProcessing/forces/0/forceCoeffs.dat")
	# flow = numpy.fromfile(example+"/postProcessing/sampleDict/8000/allInOne.raw", dtype=numpy.float32)
	# flow = numpy.reshape(flow, [192, 128, 6])
	os.system("cp "+example+"/postProcessing/sampleDict/8000/allInOne.raw"+" valSet/"+str(index)+".raw")
	groundTruth[index] = forces[-1][2:4]
	valSetToCasesMap[index] = example
	index += 1

pickle.dump(valSetToCasesMap, open("valSetToCasesMap.pkl", "wb"))
numpy.save("valGroundTruth.npy", groundTruth)

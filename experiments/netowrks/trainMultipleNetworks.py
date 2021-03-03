#Train same setup multiple times
import sys
import os
from multiprocessing import Pool


def trainNets(inpt):
	gpu = inpt % 4
	run = inpt
	modelTypes = ["vc"]
	# modelTypes = ['vc', "ds", "op", "cc"]
	# if run > 12:
	# 	modelTypes = ['vc', "op", "cc"]
	# else:
	# 	modelTypes = ["op", "cc"]
	for modelType in modelTypes:
		os.system("CUDA_VISIBLE_DEVICES="+str(gpu)+" python3 trainOnMultipleGPUs.py "+modelType+" "+str(run))

for s in range(1):
	print("Doing s="+str(s))
	runs = [i for i in range(s*4+1,s*4+5)]
	p = Pool(4)
	res = p.map(trainNets, runs)
	p.close()
	p.join()

print("Done!")

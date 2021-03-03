#Run test network scripts on repetative runs, in parallel
import os
import sys
from multiprocessing import Pool

# modelTypes = ["op", "cc"]
# modelTypes = ["vc", "ds", "op", "cc"]
modelTypes = ["vc"]
numRuns = 4
basePath = sys.argv[1]
mType = sys.argv[2]
if basePath[-1] != "/":
	exit("Path must end with a slash")
# gpu = sys.argv[1]
# releaseDirs = ["vc/1/","vc/2/","vc/3/","vc/4/"]

def runTest(gpu):
	run = str(gpu+1)
	relDir = basePath+run+"Release/"
	if not os.path.isdir(relDir):
		print(relDir)
		return
	# os.system('python3 testNetworksOnFlow.py '+relDir+" "+mType)
	# os.system('CUDA_VISIBLE_DEVICES='+str(gpu)+' python3 testNetworksOnFlow.py '+relDir+" "+mType)
	os.system('CUDA_VISIBLE_DEVICES='+str(gpu)+' python3 testNetworks.py '+relDir+" "+mType)


runs = [i for i in range(4)]
p = Pool(4)
res = p.map(runTest, runs)
p.close()
p.join()


# for mType in modelTypes:
# 	for run in range(numRuns):
# 		# relDir = basePath+mType+"/"+str(run+1)+"/"
# 		relDir = basePath+str(run+1)+"Release/"

# 		if not os.path.isdir(relDir):
# 			print(relDir)
# 			continue
# 		os.system('CUDA_VISIBLE_DEVICES='+gpu+' python3 testNetworks.py '+relDir+" "+mType)
# 		# os.system('python3 testNetworks.py '+relDir+" "+mType)
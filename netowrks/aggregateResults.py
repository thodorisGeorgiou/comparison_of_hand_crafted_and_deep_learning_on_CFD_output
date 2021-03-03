#Aggregate results for a setup. To be run from within the setup directory
import sys
import os
import re
import numpy

# types = ["op", "cc"]
types = ["ds", "vc", "op", "cc"]

drag = []
lift = []
dragSTD = []
liftSTD = []
for t in types:
	drag.append(0)
	lift.append(0)
	dragSTD.append(0)
	liftSTD.append(0)
	runs = os.listdir(t)
	count = 0
	for r in runs:
		if "Release" not in r:
			continue
		elif (t == "op" or t=="cc") and ("9" in r or "10" in r or "11" in r or "12" in r):
			continue
		path = t+"/"+r+"/Accuracy4GPUHBS_SQR.txt"
		try:
			f = open(path, "r")
		except FileNotFoundError:
			print(path)
			continue
		count += 1
		res = f.read()
		res = re.split("[| |]| - ", res)
		try:
			drag[-1] += float(res[0][1:])
		except ValueError:
			print(path)
			print(r)
			exit()
		print(float(res[0][1:]), end="\t")
		dragSTD[-1] += float(res[0][1:])**2
		i = 1
		while True:
			if res[i] == '':
				i += 1
			else:
				lift[-1] += float(res[i][:-1])
				liftSTD[-1] += float(res[i][:-1])**2
				print(float(res[i][:-1]))
				break
		# print(i)
	# print("Count = "+str(count))
	drag[-1] /= count
	lift[-1] /= count
	dragSTD[-1] /= count
	liftSTD[-1] /= count

print(types)
print(drag)
print(dragSTD)
print(lift)
print(liftSTD)

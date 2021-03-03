import os
import sys
import numpy
import pickle
import sklearn.cluster
import sklearn.ensemble

#Set path for opencv library
cvPath = "/scratch/georgioutk/opencv/install/lib/python3.6/dist-packages"

#Define the path to load the descriptors
descriptor = "HL_ORB/"

#Load more libraries
sys.path.append(cvPath)
import cv2
import preprocessing
from matplotlib import pyplot as plt

#Load flow fields.
[trainSet, trainLabels], [testSet, testLabels] = preprocessing.loadTrainAndTestSets()

#Define detector and descriptor
descr = cv2.ORB_create()
descr.setFastThreshold(0)
descr.setEdgeThreshold(0)
# descr = cv2.xfeatures2d.SIFT_create()
# descr = cv2.xfeatures2d.SURF_create()
# det = cv2.xfeatures2d.SIFT_create()
# det = cv2.xfeatures2d.HarrisLaplaceFeatureDetector_create()
# det = cv2.AgastFeatureDetector_create()
# det = cv2.ORB_create()
# det.setFastThreshold(0)
# det.setEdgeThreshold(0)

det = cv2.xfeatures2d.HarrisLaplaceFeatureDetector_create()
# det = cv2.xfeatures2d.StarDetector_create()

#If descriptor file exists do nothing, else extract and save.
print("Extracting train features")
try:
	descriptors = pickle.load(open(descriptor+"combinedTrainDescriptors.pkl", "rb"))
except FileNotFoundError:
	count = 0
	descriptors = []
	for im in trainSet:
		count += 1
		if count % 100 == 0:
			print(count/15000, end="\r", flush=True)
		kp = det.detect(im[:,:,0])
		for i in range(1,5):
			nkp = det.detect(im[:,:,i])
			toGo = []
			for k in range(len(nkp)):
				for kk in kp:
					if kk.overlap(kk, nkp[k]) > 0.9:
						toGo.append(k)
						break
			for k in toGo[::-1]:
				del(nkp[k])
			kp += nkp
		descs = []
		if len(kp) == 0:
			continue
		for i in range(5):
			d = descr.compute(im[:,:,i], kp)[1]
			descs.append(d)
		descs = numpy.concatenate(descs, axis=1)
		descriptors.append(descs)
	pickle.dump(descriptors, open(descriptor+"combinedTrainDescriptors.pkl", "wb"))
	descriptors = numpy.concatenate(descriptors, axis=0)

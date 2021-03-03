#Code to make plots for paper
import numpy
from matplotlib import pyplot as plt

def rSquared(preds, gt):
	ssGt = numpy.sum(numpy.square(gt - numpy.mean(gt)), axis=0)
	ssPreds = numpy.sum(numpy.square(preds - numpy.mean(gt)), axis=0)
	return ssPreds/ssGt

gt = numpy.load("vcGt2.npy")
sift = numpy.load("siftPredsNew.npy")
vc = numpy.load("vcPredictions2.npy")
vcRF = numpy.load("vcRF2Predictions2.npy")


liftInds = numpy.argsort(gt[:,1])
dragInds = numpy.argsort(gt[:,0])
plt.plot(gt[liftInds, 1]-sift[liftInds, 1], color="red")
plt.plot(gt[liftInds, 1]-vc[liftInds, 1], color="blue")
plt.plot(gt[liftInds, 1]-vcRF[liftInds, 1], color="green")

plt.plot(numpy.sort(numpy.abs(gt[liftInds, 1]-sift[liftInds, 1])), color="red", label="DE-SIFT-MD")
plt.plot(numpy.sort(numpy.abs(gt[liftInds, 1]-vc[liftInds, 1])), color="blue", label="VC")
plt.plot(numpy.sort(numpy.abs(gt[liftInds, 1]-vcRF[liftInds, 1])), color="green", label="VC-RF")
plt.legend(loc="upper left", fontsize="xx-large")
plt.savefig("liftTest")
# plt.show()

plt.plot(numpy.sort(numpy.abs(gt[dragInds, 0]-sift[dragInds, 0])), color="red", label="DE-SIFT-MD")
plt.plot(numpy.sort(numpy.abs(gt[dragInds, 0]-vc[dragInds, 0])), color="blue", label="VC")
plt.plot(numpy.sort(numpy.abs(gt[dragInds, 0]-vcRF[dragInds, 0])), color="green", label="VC-RF")
plt.legend(loc="upper left", fontsize="xx-large")
plt.savefig("dragTest")
plt.show()

rSquared(sift, gt)
rSquared(vc, gt)
rSquared(vcRF, gt)

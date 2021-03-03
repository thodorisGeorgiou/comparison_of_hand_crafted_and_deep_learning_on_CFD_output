import sys

#templatePath = "/tank/georgioutk/latopCleanUp/Documents/dataSetHelpers/extractSlicesNew/sampleDict"
templatePath = "/tank/airfoil/postProcessScripts/sampleDict"
fileName = sys.argv[1]

res = 64
file = open(templatePath, "r")
cont = file.readlines()
file.close()

resolution = [128+64, 1, 128]
mmin = [0.8, 0, -1]
mmax = [3.8, 0, 1]
step = [float(mmax[0] - mmin[0])/resolution[0], float(mmax[1] - mmin[1])/resolution[1], float(mmax[2] - mmin[2])/resolution[2]]

points = ""
for i in range(resolution[0]):
	x = mmin[0]+step[0]*i
	for j in range(resolution[1]):
		y = mmin[1]+step[1]*j
		for k in range(resolution[2]):
			z = mmin[2]+step[2]*k
			points += "("+str(x)+" "+str(y)+" "+str(z)+")"

newCont = ""
for row in range(len(cont)):
	if "res" in cont[row]:
		newRow = cont[row][:-1]+"Airfoil\n"
		newCont += newRow
	elif "points" in cont[row]:
		newRow = cont[row][:-3]+points+");\n"
		newCont += newRow
	else:
		newCont += cont[row]

file = open(fileName, "w")
file.write(newCont)
file.close()

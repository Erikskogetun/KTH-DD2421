import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

global PMatrix
global inputs
global targets

### TO - DO
# 1 - XC funktionen
# 2 - Zerofun
# 3 - Objective func

def objective(alpha):
	arrayLength = len(alpha)
	print(alpha)
	print(len(alpha))
	print("#######################################")
	#numpy.dot(PMatrix)
	# PMatrix finns redan här

	tempo_vector = numpy.dot(alpha,Pmatrix) # Ger oss en vektor med längden N
	tempo_scalar = numpy.dot(alpha,tempo_vector) # Ger oss en scalar
	right_scalar = numpy.sum(alpha)

	scalarValue = 1/2*tempo_scalar - right_scalar

	scalarValue = 0
	return scalarValue

# Takes a vector and returns a scalar value
# Calculate value which should constraints (tvinga)
# ti = target class (-1 or 1)
def zerofun(vector):
	#for i in range(len(targets)):
		#temp = targets[i]*vector[i]
		#scalar = temp+scalar

	numpy.dot(target,vector) # ska vara lika med noll
	return scalar

def linearKernal(dp1, dp2):
	return numpy.dot(dp1, dp2) + 1


def polynomialKernal(dp1, dp2, power = 3):
	return numpy.power((numpy.dot(dp1, dp2) + 1), power)


def radial_kernel(dp1, dp2, sigma=2):
    diff = numpy.subtract(dp1, dp2)
    return math.exp((-numpy.dot(diff, diff)) / (2 * sigma * sigma))


def calculatePMatrix():
	global PMatrix
	PMatrix = numpy.zeros((len(inputs),len(inputs)), dtype=numpy.double)
	global kernalValue
	kernalValue = numpy.zeros((len(inputs),len(inputs)), dtype=numpy.double)

	for i in range(len(inputs)):
		for j in range(len(inputs)):
			PMatrix[i][j] = targets[i]*targets[j]*linearKernal(inputs[i], inputs[j])


################################################################################
#Generating data
################################################################################
def generateData():
	numpy.random.seed(100)

	classA = numpy.concatenate((numpy.random.randn(10, 2) * 0.2 + [1.5, 0.5], numpy.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))
	classB = numpy.random.randn(20, 2) * 0.2 + [0.0 , -0.5]

	inputs = numpy. concatenate ((classA, classB))
	targets = numpy.concatenate((numpy.ones(classA.shape[0]), - numpy.ones(classB.shape[0])))

	N = inputs.shape[0] # Number of rows (samples)

	permute = list(range(N))
	random.shuffle(permute)
	inputs = inputs[permute, : ]
	targets = targets[permute]

	return N, inputs, targets
################################################################################




def main():
	#N is number of datapoints
	#Inputs is are the x,y coordinates to the datapoints
	#Targets is the classification of the datapoints
	global inputs
	global targets
	N, inputs, targets = generateData()
	print("N:	", N)
	print("TYPE OF N:	", type(N))
	print("INPUTS:	", inputs)
	print("TARGETS:	", targets)
	print("LENGTH OF INPUTS", len(inputs))
	print("LENGTH OF TARGETS", len(targets))


	calculatePMatrix()
	global PMatrix
	start = numpy.zeros(N) #Initial guess
	B = [(0, None) for b in range(N)]
	# B = [(0, C) for b in range(N)]
	XC = {'type':'eq', 'fun':zerofun} # Sumi = ai*ti = 0
	# Avvaktar lite med XC

	for i in range(N):
		ret = minimize(objective, start, bounds=B, constraints=XC)
		x=0	#Vilket värde vill vi ha?
		alpha = ret[x]
		alphaValues[i] = alpha


if __name__ == "__main__":
	main()

import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

global PMatrix
global inputs
global targets
global classA
global classB


def objective(alpha):
	global PMatrix
	tempo_vector = numpy.dot(alpha,PMatrix) # Ger oss en vektor med längden N
	tempo_scalar = numpy.dot(alpha,tempo_vector) # Ger oss en scalar
	right_scalar = numpy.sum(alpha)
	scalarValue = 1/2*tempo_scalar - right_scalar

	return scalarValue


def zerofun(alpha):
	global targets
	return numpy.dot(targets,alpha)


def indicatorFunction(dp1, dp2):
	global 
	print("DP1:	", dp1)
	print("DP2:	", dp2)

	indication =

	return


def linearKernal(dp1, dp2):
	return numpy.dot(dp1, dp2) + 1


def polynomialKernal(dp1, dp2, power = 3):
	return numpy.power((numpy.dot(dp1, dp2) + 1), power)


def radial_kernel(dp1, dp2, sigma=2):
    diff = numpy.subtract(dp1, dp2)
    return math.exp((-numpy.dot(diff, diff)) / (2 * sigma * sigma))


def calculatePMatrix():
	global PMatrix
	global kernalValue

	PMatrix = numpy.zeros((len(inputs),len(inputs)), dtype=numpy.double)
	kernalValue = numpy.zeros((len(inputs),len(inputs)), dtype=numpy.double)

	for i in range(len(inputs)):
		for j in range(len(inputs)):
			PMatrix[i][j] = targets[i]*targets[j]*linearKernal(inputs[i], inputs[j])


################################################################################
#Generating data
################################################################################
def generateData():
	global classA
	global classB
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


def plottingData(p):
	global classA
	global classB
	plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
	plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')

	plt.axis('equal') # Force same scale on both axes

	xgrid=numpy.linspace(-5, 5)
	ygrid=numpy.linspace(-4, 4)
	grid=numpy.array([[indicatorFunction(x, y) for x in xgrid ] for y in ygrid])
	plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
	plt.savefig('svmplot.pdf') # Save a copy in a file
	plt .show() # Show the plot on the screen



def main():
	#N is number of datapoints
	#Inputs is are the x,y coordinates to the datapoints
	#Targets is the classification of the datapoints
	global inputs
	global targets
	global PMatrix

	N, inputs, targets = generateData()
	print("N:	", N)
	print("TYPE OF N:	", type(N))
	print("INPUTS:	", inputs)
	print("TARGETS:	", targets)
	print("LENGTH OF INPUTS", len(inputs))
	print("LENGTH OF TARGETS", len(targets))

	calculatePMatrix()

	start = numpy.zeros(N) #Initial guess
	B = [(0, None) for b in range(N)]
	XC = {'type':'eq', 'fun':zerofun}

	ret = minimize(objective, start, bounds=B, constraints=XC)
	print(ret)

	plottingData(ret)


if __name__ == "__main__":
	main()

import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

C = 10000

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

def b_value():
	global b
	sv = non_zero_dp[0]
	kernalValue = linearKernal(inputs, sv)
	value = numpy.dot(non_zero_a, non_zero_t)
	numpy_product = numpy.dot(value, kernalValue)
	numpy_sum = numpy.sum(numpy_product)
	b = numpy_sum - non_zero_t[0]


def indicatorFunction(x,y):
	global targets
	global inputs
	global non_zero_a
	global non_zero_t
	global non_zero_dp
	global b

	dp = numpy.array([x,y])
	kernalValue = linearKernal(non_zero_dp, dp)
	value = numpy.dot(non_zero_a, non_zero_t)
	ind = numpy.dot(value,kernalValue)
	sum = numpy.sum(ind)
	indication = sum - b

	return indication

def extract(alpha):
	global non_zero_a, non_zero_t, non_zero_dp
	non_zero_a=[]
	non_zero_t=[]
	non_zero_dp=[]
	for i in range(len(alpha)):
		if abs(alpha[i])>10e-5:
			non_zero_a.append(alpha[i])
			non_zero_t.append(targets[i])
			non_zero_dp.append(inputs[i])


def linearKernal(dp1, dp2):
	return numpy.dot(dp1, dp2)


def polynomialKernal(dp1, dp2, power = 3):
	return numpy.power((numpy.dot(dp1, dp2) + 1), power)


def radial_kernel(dp1, dp2, sigma=2):
    diff = numpy.subtract(dp1, dp2)
    return math.exp((-numpy.dot(diff, diff)) / (2 * sigma * sigma))


def calculatePMatrix():
	global PMatrix
	PMatrix = numpy.zeros((len(inputs),len(inputs)), dtype=numpy.double)
	for i in range(N):
		for j in range(N):
			PMatrix[i][j] = targets[i]*targets[j]*linearKernal(inputs[i], inputs[j])

	#print("PMATRIX:	", len(PMatrix))


################################################################################
#Generating data
################################################################################
def generateData():
	global classA, classB, inputs, targets, N
	numpy.random.seed(100)
	classA = numpy.concatenate((numpy.random.randn(10, 2) * 0.2 + [1.5, 0.5], numpy.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))
	classB = numpy.random.randn(20, 2) * 0.2 + [0.0 , -0.5]
	inputs = numpy.concatenate((classA, classB))
	targets = numpy.concatenate((numpy.ones(classA.shape[0]), - numpy.ones(classB.shape[0])))
	N = inputs.shape[0] # Number of rows (samples)
	permute = list(range(N))
	random.shuffle(permute)
	inputs = inputs[permute, : ]
	targets = targets[permute]
################################################################################


def plottingData():
	xgrid=numpy.linspace(-5, 5)
	ygrid=numpy.linspace(-4, 4)
	grid = numpy.array([[indicatorFunction(x,y) for x in xgrid] for y in ygrid])
	plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))

	plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
	plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
	plt.axis('equal') # Force same scale on both axes
	plt.savefig('svmplot.pdf', dpi=1000) # Save a copy in a file
	plt .show() # Show the plot on the screen


def main():

	generateData()
	calculatePMatrix()
	start = numpy.zeros(N) #Initial guess
	B = [(0, C) for b in range(N)]
	XC = {'type':'eq', 'fun':zerofun}
	ret = minimize(objective, start, bounds=B, constraints=XC)
	alpha=ret["x"]
	print(ret['success'])
	print("*******************************")
	extract(alpha)
	b_value()
	plottingData()


if __name__ == '__main__':
	main()

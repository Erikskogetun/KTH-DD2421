import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt


P = 0 # P-matrisen än i*j = N*N
classA = []
classB = []
inputs = []
targets = []
N = 0
C = 10000
nonZeroAlpha = []
alpha = []
b = 0
kernel=0



def createDebugSet3():
	##för att debugga enligt 6.1
	##datap från method2 https://ai6034.mit.edu/wiki/images/SVM_and_Boosting.pdf
	##
	global classA, classB, inputs, targets, N
	classA = numpy.zeros(shape=(1,2))
	classB = numpy.zeros(shape=(1,2))
	classA[0] =  [0,0] #neg
	classB[0] =  [0,0]

	targets = numpy.asarray([-1,1])
	inputs = numpy.concatenate((classA, classB))
	#print(inputs)

	N = inputs.shape[0]


def createDebugSet2():
	##för att debugga enligt 6.1
	##datap från http://axon.cs.byu.edu/Dan/678/miscellaneous/SVM.example.pdf
	##
	global classA, classB, inputs, targets, N
	classA = numpy.zeros(shape=(4,2))
	classB = numpy.zeros(shape=(4,2))
	classA[0] =  [3,1] #pos
	classA[1] =  [3,-1] #pos
	classA[2] =  [6,1] #pos
	classA[3] =  [6,-1] #pos

	classB[0] =  [1,0]
	classB[1] =  [0,1]
	classB[2] =  [0,-1]
	classB[3] =  [-1,0]

	targets = numpy.asarray([1,1,1,1,-1,-1,-1,-1])
	inputs = numpy.concatenate((classA, classB))
	#print(inputs)

	N = inputs.shape[0]

def createDebugSet():
	##för att debugga enligt 6.1
	##datap från method2 https://ai6034.mit.edu/wiki/images/SVM_and_Boosting.pdf
	##
	global classA, classB, inputs, targets, N
	classA = numpy.zeros(shape=(2,2))
	classB = numpy.zeros(shape=(1,2))
	classA[0] =  [0,0] #neg
	classA[1] =  [1,1] #neg
	classB[0] =  [2,0]

	targets = numpy.asarray([1,1,-1])
	inputs = numpy.concatenate((classA, classB))
	#print(inputs)

	N = inputs.shape[0]
	#permute = list(range(N))
	#random.shuffle(permute)
	#inputs =inputs[permute, :]
	#print(inputs)
	#targets = targets[permute]



def createTestSet():
	#översta rutan på s8
	global classA, classB, inputs, targets, N
	numpy.random.seed(200) #fast seed
	classA = numpy.concatenate (
	(numpy.random.randn(10 , 2) * 0.2 + [2 , 0.5],
	numpy.random.randn(10 , 2) * 0.2 + [-2, 0.5]))
	#classA = numpy.random.randn(30 , 2) * 0.2 + [-0.25, 0.75]

	classB = numpy.random.randn(30,2)*0.2 + [0.0, 0.0]

	inputs = numpy.concatenate((classA, classB))
	#print(inputs)
	targets = numpy.concatenate((numpy.ones(classA.shape[0]),
								 -numpy.ones(classB.shape[0])))

	N = inputs.shape[0]
	permute = list(range(N))
	numpy.random.shuffle(permute)
	inputs =inputs[permute, :]
	targets = targets[permute]


def linear_kernel(x, y):
	#linear kernel
	return numpy.dot(numpy.transpose(x),y)

def pol_kernel(x, y):
	p=3
	return numpy.power((numpy.dot(numpy.transpose(x),y)+1), p)

def rad_kernel(x, y):
	expon=numpy.subtract(x,y)
	sigma=7
	return numpy.exp((-numpy.dot(expon, expon)) / (2*sigma*sigma))


def zerofun(alpha):
	return numpy.dot(numpy.transpose(alpha), targets)




def createP():

	global P
	P = numpy.zeros(shape=(N,N))
	for i in range(N):
		for j in range(N):
			P[i][j] = targets[i]*targets[j]*kernel(inputs[i], inputs[j])
			#print("kernel "+str(i)+str(j) +" är " +str(linear_kernel(inputs[i], inputs[j]))

	#print(P)


def objective(alpha):
	# global variable for t and K values
	sum = 0
	for i in range (N):
		for j in range(N):
			sum += alpha[i]*alpha[j]*P[i][j]
	sum = sum/2
	for i in range(N):
		sum -= alpha[i]

	return sum

##
##def objective(alpha):
##    # global variable for t and K values
##    sum = 0
##    sum = numpy.sum(numpy.dot(numpy.dot(alpha, alpha),P))
##    sum = sum/2
##    sum = sum - numpy.sum(alpha)
##    return sum


def createNonZeroAlpha():
	global nonZeroAlpha
	for i in range(alpha.shape[0]):
		if alpha[i] > 10**(-5):
			nonZeroAlpha.append([alpha[i], [inputs[i][0], inputs[i][1]], targets[i]])


def createAnySupportVector():
	return (nonZeroAlpha[0][1], nonZeroAlpha[0][2])

def calcLillaB():
	#calculates b according to eq7
	global b
	s, st = createAnySupportVector()
	#print("ST ", st)
	#print("T VALUE: ", nonZeroAlpha[0])
	for i in range(len(nonZeroAlpha)):
		#if nonZeroAlpha[i][0] < C:
			b += nonZeroAlpha[i][0]*nonZeroAlpha[i][2]*kernel(s, nonZeroAlpha[i][1])
	b = b-st





def indicator(x, y):
	sum = 0
	for i in range(len(nonZeroAlpha)):
		sum += nonZeroAlpha[i][0]*nonZeroAlpha[i][2]*kernel([x,y], nonZeroAlpha[i][1])
	print (sum -b)
	return sum - b


def plotData():

	plt.plot([p[0] for p in classA],
			 [p[1] for p in classA],
			 'b.')

	plt.plot([p[0] for p in classB],
			 [p[1] for p in classB],
			 'r.')

	plt.axis('equal') # force same scale on both axes
	name = kernel.__name__ + " C ="+ str(C) + ".png"
	plt.title(name, fontsize=12, loc='left')
	plt.savefig(name,dpi=1000) # save a copy in the file
	plt.show() # show the plot on the screen

def plotDB():
	xgrid = numpy.linspace(-5,5)
	ygrid = numpy.linspace(-4,4)

	grid = numpy.array([[indicator(x,y)
						 for x in xgrid]
						for y in ygrid])
	plt.contour(xgrid, ygrid, grid,
				(-1.0, 0.0, 1.0),
				colors = ('red', 'black', 'blue'),
				linewidths =(1,3,1))


def main():

	global alpha, kernel
	kernel=pol_kernel

	createTestSet()
	print("INPUTS TARGETS N ", inputs, targets, N)
	#createDebugSet2()
	createP()
	#linearly_separable_dataset()
	XC = {'type' : 'eq', 'fun' : zerofun}
	start = numpy.zeros(N)
	B = [(0, C) for b in range(N)] # sets lower limit
	#B = [(0, None) for b in range(N)] # sets lower limit
	ret = minimize(objective, start, bounds = B, constraints = XC)
	alpha = ret['x']
	#print(ret)
	print(ret['success'])
	#print(kernel.__name__)

	#print("alpha", alpha)
	createNonZeroAlpha()
	#print("non0a", nonZeroAlpha)
	calcLillaB()
	#print("b", str(b))
	plotDB()
	plotData()





main()

import numpy, random, math
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import kernels

# generate data ------------------------------------------------------------
def generateData(classA1_mean, classA1_cov, classA1_num, classA2_mean, classA2_cov, classA2_num, classB_mean, classB_cov, classB_num, seed = None):
    if seed != None :
        np.random.seed(seed)
    classA1 = np.random.multivariate_normal(classA1_mean, classA1_cov, classA1_num)
    classA2 = np.random.multivariate_normal(classA2_mean, classA2_cov, classA2_num)
    classA = np.concatenate((classA1,classA2))
    classB = np.random.multivariate_normal(classB_mean, classB_cov, classB_num)
    return classA,classB

def shapingData(classA,classB):
    inputs= np.concatenate((classA,classB))
    targets= np.concatenate((np.ones(classA.shape[0]),-np.ones(classB.shape[0])))    
    return inputs, targets

def shuffleData(inputs, targets):
    N=inputs.shape[0]
    permute = list(range(N))
    random.shuffle(permute)
    data= inputs[permute, :]
    t= targets[permute]
    num_samples = len(t)
    return data,t,num_samples

# functions -------------------------------------------------------
# compute values of P for objective function
def precalcKernelVal(data,num_samples,kernelFun):
    P = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(num_samples):
            P[i, j]= t[i] * t[j] * kernelFun[0](data[i], data[j],kernelFun[1])
    return P

# dual formulation function
def objective(a):
    tmp= 0.5*(a@P@a)
    scalar= tmp - np.sum(a)
    return scalar

#constraint calculation
def zerofun(a):
    return np.dot(a,t)

#identify non-zero data samples
def findNonZero(alpha):
    eps = 1e-8
    nonZeros=[]
    for idx, ai in enumerate(alpha):
        if(ai >eps):
            nonZeros.append([ai,data[idx],t[idx],idx])
    return nonZeros

#find support vectors from non-zero data samples
def findSVs(nonZeros, C):
    eps = 1e-8
    SVs=[]
    for idx, val in enumerate(nonZeros):
        if(val[0]>eps and (C == None or val[0]<C-eps)):
            SVs.append(nonZeros[idx])
    return SVs

#calculate bias
def calcBias(sv, alpha,kernelFun):
    K=numpy.zeros(num_samples)
    for idx, x in enumerate(data):
        K[idx]=kernelFun[0](sv[1],x,kernelFun[1])
    tmp = (K*t)@alpha
    #tmp = np.multiply(t,K)
    #tmp= np.multiply(alpha,tmp)
    #tmp= np.sum(tmp)

    scalar= tmp- sv[2]
    return scalar

def indicatorfun(x, nonZeros, b, kernelFun):
    ind = 0
    for nonZero in nonZeros:
        datapoint = nonZero[1]
        ind += nonZero[0]*nonZero[2]*kernelFun[0](datapoint,x,kernelFun[1])
    return ind - b

def classify(x, nonZeros, b, kernelFun):
    classification = np.zeros(num_samples)
    for i in range(num_samples):
        classification[i] = np.sign(indicatorfun(data[i], nonZeros,b,kernelFun))
    return classification
    
def accuracy(t,pred):
    ncorrect = np.sum(t==pred)
    acc = ncorrect/num_samples
    print("Accuracy on training data :",acc)
    return acc
    
# training ----------------------------------------------------------

def minimization(C,num_samples):
    start = np.random.randn(num_samples)
    B= [(0, C) for b in range(num_samples)]
    XC= {'type':'eq', 'fun':zerofun}
    
    ret= minimize(objective, start, bounds=B, constraints=XC) # finds vector which minimizes function objective within bounds and constraints
    alpha = ret['x']
    
    tmp= ret['success']
    if(ret['success']==False):
        raise Exception("Minimization failed")
    print("Minimization succeeded")
    
    return alpha

def extract(alpha, kernelFun, C):
    nonZeros = findNonZero(alpha)
    svs = findSVs(nonZeros, C)
    if len(svs) < 3:
        raise Exception("Not enough support vectors")
    b = calcBias(svs[0], alpha, kernelFun)
    return nonZeros,svs,b

# plot data -----------------------------------------------------------------

def plotResults(classA,classB,svs,b,kernelFun):
    plt.figure(figsize = (20,15))
    plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
    plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
    plt.plot([sv[1][0] for sv in svs], [sv[1][1] for sv in svs], 'ko')
    plt.tick_params(axis = 'both', which = 'both',labelsize = 30)
    plt.axis('equal')
    plt.grid()
    n = 100
    xgrid= np.linspace(-5,5,n)
    ygrid=np.linspace(-4,4,n)
    grid=np.array([[indicatorfun([x,y],nonZeros, b, kernelFun)
                    for x in xgrid]
                    for y in ygrid])
    plt.contour(xgrid,ygrid,grid,(-1.0,0.0,1.0),
                colors=('red', 'black', 'blue'),
                linewidths=(1,3,1))
    
    plt.savefig('svmplot.pdf')
    plt.show()



# TESTING ##############################################################

# Data generation
classA, classB = generateData([ 1.5, -0.5],[[0.5**2,0.0],[0.0,0.5**2]],10,
                              [-1.5, 0.5],[[0.5**2,0.0],[0.0,0.5**2]],10,
                              [ 0.0,-0.5],[[0.2**2,0.0],[0.0,0.2**2]],20, seed=100)
inputs, targets = shapingData(classA,classB)
data,t,num_samples = shuffleData(inputs,targets)

# Parametrization of the problem
C = 1     # C = None is the same as C = inf
#kernelFun = [kernels.linearK,None]
kernelFun = [kernels.polynomialK,3]
#kernelFun = [kernels.radialK,1.0]

# Training model
P = precalcKernelVal(data,num_samples,kernelFun)   
alpha = minimization(C,num_samples)
nonZeros,svs,b = extract(alpha,kernelFun,C)

# Classification
classification = classify(data, nonZeros, b, kernelFun)
acc = accuracy(t,classification) 

# Plotting
plotResults(classA,classB,svs,b,kernelFun)
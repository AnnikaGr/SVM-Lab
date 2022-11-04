import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def linearK(xi,xj,param):
    return numpy.dot(xi,xj)

def polynomialK(xi,xj,p):
    return numpy.power(numpy.dot(xi,xj)+1,p)

def radialK(xi,xj,sigma):
    comp = numpy.exp(-numpy.linalg.norm(xi-xj)**2/(2*sigma**2))
    return comp


# TODO implement other kernel functions with order
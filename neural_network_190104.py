# -*- coding: utf-8 -*-
"""
Neural Network 

Version 190104



Neural network project: maximal HAZ hardness prediction

@author: Freenox
"""

#Libraries

import numpy as np
import matplotlib.pyplot as plt


#Extraction of input values

values = np.loadtxt("welding_data.txt")
X = np.zeros((len(values),5))
y = np.zeros((len(values),1))

for i in range(len(values)):
    X[i] = values[i][:5]
    y[i] = values[i][-1]


#Scaling

Xmax = [1,1,200,0.2,0.4]
Ymax = [300]
X = X / Xmax
y = y / Ymax



class NeuralNetwork(object):
    def __init__(self,sizes):
        """Creation of a neural network with [sizes] dimensions"""

        self.numberLayers = len(sizes)
        
        #Initialization of weights and biases
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]
        

    def forward(self,a):
        """Return result of NN for input a"""

        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a


    def cost(self,output,y):
        """Return the cost of the outputs"""

        return np.mean(np.square(output - y))


    def costDerivative(self,output, y):
        """Return the cost derivative function of outputs"""

        return (output - y)


    def backPropagation(self, x, y):
        """Return the gradients matrices of the cost function for weights and biases"""
    
        #Initialization of gradient matrices
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        #Creating list of all activations per layer
        activation = x
        activations = [x] #list to store activation vectors by layer
        zs = [] #list to store z vectors by layer
        for b,w in zip(self.biases, self.weights):
        	z = np.dot(w*activation+b)
        	zs.append(z)
        	activation = sigmoid(z)
        	activations.append(activation)
        
        #First occurence of gradient matrices on last layer
        delta = self.costDerivative(activations[-1],y)*derivateSigmoid(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        	#Iterative update
        for l in range(2, self.numberLayers):
        	z = zs[-l]
        	spz = derivateSigmoid(z)
        	delta = np.dot(self.weights[-l+1].transpose(), delta)*spz
        	nabla_b[-l] = delta
        	nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)


    def update(self, X, Y, eta):
        """Updates the weights and biases with a learning rate eta"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in zip(X,Y):
            delta_nabla_b, delta_nabla_w = self.backPropagation(X,Y)
            nabla_b = [nb + dnb for nb,dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw,dnw in zip(nabla_w, delta_nabla_w)]
            	
        self.weights = [w + (eta*nw) for w,nw in zip(self.weights, nabla_w)]
        self.biases = [b + (eta*nb) for b,nb in zip(self.biases, nabla_b)]
	

#Definition of auxiliary functions

def sigmoid(self, x):
    return 1/(1+np.exp(-x))

def derivateSigmoid(self, x):
    return self.sigmoid(x)*(1-self.sigmoid(x))


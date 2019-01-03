# -*- coding: utf-8 -*-
"""
Neural Network 

Created on 2018-12-25 21:47:12
Updated on XXXX-XX-XX XX:XX:XX

Neural network project: maximal HAZ hardness prediction

Input: Root_HI; Hot_HI; Preheat; PCM; CE [5]
Hidden Layers: 1
Output: Max HAZ hardness [1]

@author: Freenox
"""
import numpy as np
import matplotlib.pyplot as plt

"""VALEURS
Les valeurs sont tirees du fichier texte joint
#X: (HI_R, HI_H, Temp, PCM, CE)
#y: (HV_MAX_ROOT_HAZ)
"""
values = np.loadtxt("welding_data.txt")
X = np.zeros((len(values),5))
y = np.zeros((len(values),1))

for i in range(len(values)):
    X[i] = values[i][:5]
    y[i] = values[i][-1]


xPredicted = np.array(([0.8,0.8,150,0.170,0.360],
                       [0.1,0.1,10,0.170,0.40]), dtype=float)

"""Highest values are set"""

Xmax = [1,1,200,0.2,0.4]
Ymax = [300]
X = X / Xmax
y = y / Ymax
xPredicted = xPredicted / Xmax


class NeuralNetwork(object):
    def __init__(self,sizes):
        """ Creation d'un reseau avec les les dimensions de la liste size"""
        self.numberLayers = len(sizes)
        
        #Initialisation des poids et biais
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]
        
    def forward(self,a):
        """Propagation"""
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

#    
#    def backward(self,X,y,o):
#        """Methode de descente de gradient"""
#        #Calcul des erreurs et des gradients
#        self.output_error = y-o
#        self.output_variation = self.output_error*self.derivateSigmoid(o)
#        
#        self.HL_error = self.output_variation.dot(self.W2.T)
#        self.HL_variation = self.HL_error*self.derivateSigmoid(self.z)
#        
#        #Mise à jour des matrices de poids
#        self.W1 += X.T.dot(self.HL_variation)
#        self.W2 += self.z.T.dot(self.output_variation)
#
#    def train(self,X,y):
#        self.backward(X,y,self.forward(X))
#        
#    def predict(self,xpredicted):
#        print("Predicted data based on trained weights: ")
#        print("Input: \n" + str(xpredicted*Xmax))
#        print("Output: \n" + str(self.forward(xpredicted)*Ymax))
#        
def sigmoid(self, x):
    return 1/(1+np.exp(-x))

def derivateSigmoid(self, x):
    return self.sigmoid(x)*(1-self.sigmoid(x))
#NN = NeuralNetwork(5,10,1)
#loss = []
#
#
#for i in range(1000):  # Training
#    loss.append(np.mean(np.square(y - NN.forward(X))))
#    NN.train(Xprime, yprime)
#    
#print(" #" + str(i) + "\n")
#print("Input (scaled): \n" + str(Xprime))
#print("Actual Output: \n" + str(yprime))
#print("Predicted Output: \n" + str(NN.forward(Xprime)))
#print("Loss: \n" + str(np.mean(np.square(yprime - NN.forward(Xprime)))))  # mean sum squared loss
#print("\n")
#
#plt.plot(loss)
#plt.show()
#NN.predict(xPredicted)


    
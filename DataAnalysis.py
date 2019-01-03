# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 22:06:02 2019

@author: Freenox
"""

import numpy as np

values = np.loadtxt("welding_data.txt")
X = np.zeros((len(values),5))
Y = np.zeros((len(values),1))

for i in range(len(values)):
    X[i] = values[i][:5]
    Y[i] = values[i][-1]
    
print(X,Y)
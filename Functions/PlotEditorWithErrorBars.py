#Importing
import os
import sys
import numpy as np

sys.path.insert(0,"Functions")
from Plotters import *


#Retrieve the data

#Frequency arrays
Frequencies = np.genfromtxt("Data/Frequencies.csv",delimiter=",")

#Eigenvalue arrays
Eigenvalues = np.genfromtxt("Data/Eigenvalues.csv",delimiter=",",dtype=complex)

#Tensor coefficient arrays
Tensors = np.genfromtxt("Data/Tensors.csv",delimiter=",",dtype=complex)

#Error Bars
Errors = np.genfromtxt("Data/ErrorBars.csv",delimiter=",")



#remove the rows so that the array represents an upper triangular matrix
Tensors = np.concatenate([np.concatenate([Tensors[:,:3],Tensors[:,4:6]],axis=1),Tensors[:,8:9]],axis=1)
Errors = np.concatenate([np.concatenate([Errors[:,:3],Errors[:,4:6]],axis=1),Errors[:,8:9]],axis=1)

#define the place to store it This is relative to the current file
savename = "Graphs/"

#plot the graphs
Show = EigPlotter(savename,Frequencies,Eigenvalues)
Show = ErrorPlotter(savename,Frequencies,Tensors,Errors)

#plot the graph if required
if Show==True:
    plt.show()
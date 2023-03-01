#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 08:09:55 2023

@author: chloe
"""

""" PCA through Singular Value Decomposition """
print(" ----------- PCA through Singular Value Decomposition ----------- ")

import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
# Defined 3 points in 2D-space:
X=np.array([[2, 1, 0],[4, 3, 0]])

# Calculate the covariance matrix: (# R = np.cov(X, bias=True))
Y = np.transpose(X)     # transposé de X
R = 1/3 * np.matmul(X,Y)    # calcul de R
print("Covariance matrix R :", R)
sn.heatmap(R, annot=True, fmt='g')
plt.show()

# Calculate the SVD decomposition and new basis vectors:
[U,D,V]=np.linalg.svd(R)  # call SVD decomposition
u1=U[:,0] # new basis vectors
u2=U[:,1]

# Calculate the coordinates in new orthonormal basis:
Xi1=np.matmul(np.transpose(X), u1)
Xi2=np.matmul(np.transpose(X), u2)
print("Xi1 :", Xi1)  
print("Xi2 :", Xi2)  

# print((Xi1[:, None])) # add second dimension to array and test it
Xaprox=np.matmul(u1[:, None], Xi1[None, :]) # + np.matmul(u2[:, None], Xi2[None, :])
print("Xaprox :", Xaprox)  

# Calculate the approximation of the original from new basis
print(Xi1[:,None]) # add second dimention to array and test it

# Check that you got the original


print(" ")
""" PCA on Iris data """
print(" ----------- PCA on Iris data ----------- ")
# original data = X
# 1) pre-processing
# 2) Appl PCA on pre-processing data
# 3) use KNN-classifier on 3 types of data
#     a) original X -4 dim
#     b) data after PCA X pca
#     c) X wrong chosen 2 dimensions


# Load Iris dataset as in the last PC lab:
from sklearn.datasets import load_iris
iris=load_iris()
iris.feature_names
print("Iris feature names :", iris.feature_names)
print("Iris data :", iris.data[0:5,:])
print("Iris target :", iris.target[:])


# We have 4 dimensions of data, plot the first three colums in 3D
X=iris.data
y=iris.target
import matplotlib.pyplot as plt
plt.figure()
axes1=plt.axes(projection='3d')
axes1.scatter3D(X[y==0,1],X[y==0,1],X[y==0,2],color='green')
axes1.scatter3D(X[y==1,1],X[y==1,1],X[y==1,2],color='blue')
axes1.scatter3D(X[y==2,1],X[y==2,1],X[y==2,2],color='magenta')
plt.show

# Pre-processing is an important step, you can try either StandardScaler (zero mean, unit variance of features)
# or MinMaxScaler (to interval from 0 to 1)
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
Xscaler = StandardScaler()
Xpp=Xscaler.fit_transform(X)
# print(np.mean(Xpp[:, 0]))
# print(np.std(Xpp[:, 0]))
# print(np.mean(X[:, 0]))
# print(np.std(X[:, 0]))

# define PCA object (three components), fit and transform the data
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(Xpp)
Xpca = pca.transform(Xpp)
print("PCA covariance", pca.get_covariance())
# you can plot the transformed feature space in 3D:
plt.figure()
axes2=plt.axes(projection='3d')
axes2.scatter3D(Xpca[y==0,0],Xpca[y==0,1],Xpca[y==0,2],color='green')
axes2.scatter3D(Xpca[y==1,0],Xpca[y==1,1],Xpca[y==1,2],color='blue')
axes2.scatter3D(Xpca[y==2,0],Xpca[y==2,1],Xpca[y==2,2],color='magenta')
plt.show

# # Compute pca.explained_variance_ and pca.explained_cariance_ratio_values
pca.explained_variance_
print("PCA variance", pca.explained_variance_)

pca.explained_variance_ratio_
print("PCA variance ratio", pca.explained_variance_ratio_)

# Plot the principal components in 2D, mark different targets in color
plt.figure()
plt.scatter(Xpca[y==0,0], Xpca[y==0,1], color='red')
plt.scatter(Xpca[y==1,0], Xpca[y==1,1], color='blue')
plt.scatter(Xpca[y==2,0], Xpca[y==2,1], color='magenta')
plt.show

print(" ")
""" KNN classifier """
print(" ----------- KNN classifier ----------- ")

# Import train_test_split as in last PC lab, split X (original) into train and test, train KNN classifier on full 4-dimensional X
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
print(" X train shape", X_train.shape)
print("X test shape", X_test.shape)

from sklearn.neighbors import KNeighborsClassifier
knn1=KNeighborsClassifier(n_neighbors = 3)
knn1.fit(X_train, y_train)
Ypred=knn1.predict(X_test)

# Import and show confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
cm = confusion_matrix(y_test, Ypred)
print("Confusion matrix", cm)
Confmat = ConfusionMatrixDisplay(confusion_matrix=cm)
Confmat.plot()
plt.show()

# Now do the same (data set split, KNN, confusion matrix), but for PCA-transformed data (1st two principal components, i.e., first two columns). 
# Compare the results with full dataset

from sklearn.model_selection import train_test_split

X_trainPCA, X_testPCA, y_trainPCA, y_testPCA = train_test_split(Xpca,y,test_size=0.3)
print(" X train shape", X_train.shape)
print("X test shape", X_test.shape)

from sklearn.neighbors import KNeighborsClassifier
knn1=KNeighborsClassifier(n_neighbors = 3)
knn1.fit(X_trainPCA, y_trainPCA)
YpredPCA=knn1.predict(X_testPCA)

# Import and show confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
cmPCA = confusion_matrix(y_testPCA, YpredPCA)
print("Confusion matrix", cm)
ConfmatPCA = ConfusionMatrixDisplay(confusion_matrix=cmPCA)
ConfmatPCA.plot()
plt.show()

# Now do the same, but use only 2-dimensional data of original X (first two columns)
from sklearn.model_selection import train_test_split

X_trainWrong, X_testWrong, y_trainWrong, y_testWrong = train_test_split(X[:, 0:1],y,test_size=0.3)
print(" X train shape", X_trainWrong.shape)
print("X test shape", X_testWrong.shape)

from sklearn.neighbors import KNeighborsClassifier
knn1=KNeighborsClassifier(n_neighbors = 3)
knn1.fit(X_trainWrong, y_trainWrong)
YpredWrong=knn1.predict(X_testWrong)

# Import and show confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
cm = confusion_matrix(y_testWrong, YpredWrong)
print("Confusion matrix", cm)
Confmat = ConfusionMatrixDisplay(confusion_matrix=cm)
Confmat.plot()
plt.show()












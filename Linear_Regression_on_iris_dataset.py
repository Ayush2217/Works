import pandas as pd
import numpy as np
from sklearn.decomposition import PCA as pca
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
import pdb
from mpl_toolkits.mplot3d import Axes3D
from sklearn.utils import shuffle as shuffled
from tkinter import *    
from tkinter import messagebox


class perceptron:
    def __init__(self, epoch, lr=1):
        # Initialization of variables
        self.epoch = epoch
        self.w = np.zeros(len(X.T))
        self.lr = lr
        self.error = []

    def ol(self, X, Y, shuffle: bool = False, random_initalization: bool = False):
        # Implementing the Online ALgorithm
        if random_initalization:
            self.w = np.random.randn(len(X.T))
        for epochs in range(self.epoch):
            if shuffle:
                X, Y = shuffled(X, Y)
            i = 0
            while i <= len(X) - 1:
                if ((y_lable[i] == 1) and ((np.dot(self.w, X[i]))) >= 0) or (
                        (y_lable[i] == -1) and ((np.dot(self.w, X[i]))) < 0):
                    i += 1
                elif ((y_lable[i] == 1) and ((np.dot(self.w, X[i]))) <= 0):
                    self.w = self.w + self.lr * X[i]
                else:
                    self.w = self.w - self.lr * X[i]
                i += 1
            self.error.append(np.linalg.norm(Y - np.where(np.dot(X, self.w) > 0, 1, -1)))

    def bl(self, X, Y, shuffle: bool = False, random_initalization: bool = False):
        if random_initalization:
            self.w = np.random.randn(len(X.T))
        for epochs in range(self.epoch):
            if shuffle:
                X, Y = shuffled(X, Y)
            delta_j = lr * np.dot((Y - np.where(np.dot(X, self.w) > 0, 1, -1)), X)
            self.w = self.w + delta_j
            self.error.append(np.linalg.norm(Y - np.where(np.dot(X, self.w) > 0, 1, -1)))
# ---- function for labling the classes --- #
def lable(Y, num, y_lable, y_classes):
    # --- y_classes have the classes to be classified in --- #
    for i in range(len(Y)):
        if Y[i] == y_classes[num]:
            y_lable.append(1)  # if tru class then we append 1 else we give -1
        else:
            y_lable.append(-1)

    return y_lable


# -----------------------------------------------------------#


# --- reading csv files --- #
data = pd.read_csv('iris.data', sep=",")
data_1 = np.array(data)

# -----------------------------------------------------------#
ui_1 = int(input('Do you want data to be shuffled, if yes enter 1'))
ui_2 = int(input('Do you want weights to be randomaly initialized, if yes enter 1'))
lr_q = int(input('Do you want your own learning rate, if yes enter 1'))
if_batch = int(input('For batch mode enter 1'))
lr = 1
if lr_q == 1:
    lr = float(input('Enter the learning rate'))
for ite in range(3):
    axes = Axes3D(plt.figure(1, figsize=(8, 6)), elev=-150, azim=30)  # Axes creation

    # -----------------------------------------------------------#

    # --- getting the features from the data above --- #
    new = [[5.1, 3.5, 1.4, 0.2, 'Iris-setosa']]
    for i in range(0, len(data_1)):
        new.append((data_1[i]))
    new = np.array(new)
    ui_2 = 2
    X_data = new[:, 0:4]
    XX = []
    for i in range(len(X_data)):
        y = []
        for j in range(len(X_data[1])):
            y.append(X_data[i][j])
        XX.append(y)
    XX = np.array(XX)


    # -----------------------------------------------------------#

    # PCA Extraction #
    # --- PCA extraction not using module can be used --- #
    def pca_without_module(XX):
        feat = np.array(XX.T)
        cov_mat = np.cov(feat)
        evals, evecs = np.linalg.eig(cov_mat)
        X = np.array(XX).dot(evecs[0:3, :].T)
        return X
        # --- PCA extraction using sklearn --- #


    X = pca(n_components=3).fit_transform(XX)
    # -----------------------------------------------------------#

    # --- scatter plot of data points --- #
    Y = new[:, 4]
    y_classes = list(set(list(Y)))
    y_labled = []
    for i in range(len(Y)):
        for j in range(len(y_classes)):
            if Y[i] == y_classes[j]:
                # defining lables as 1, 2, 3
                y_labled.append(j)
    axes.scatter(X[:, 0], X[:, 1], X[:, 2], c=np.array(y_labled),
                 cmap=plt.cm.Set1, edgecolor='k', s=40)
    # -----------------------------------------------------------#

    # --- labling data points such that we can do one vs all --- #

    y_lable = lable(Y, ite, [], y_classes)

    # -----------------------------------------------------------#

    X = np.c_[np.ones(len(X)), X]  # appending ones to incorporate bias

    # -----------------------------------------------------------#

    # --- weights calculation --- #
    flag = 0
    n = 1
    # --- different eppochs for different cases has been taken to show the hyperplane better --- #
    epoch = 10
    if (ui_1 == 1) and (ui_2 != 1):
        epoch = 70
        model = perceptron(epoch=epoch, lr=lr)
        if if_batch == 1:
            model.bl(X, y_lable, shuffle=True, random_initalization=False)
        else:
            model.bl(X, y_lable, shuffle=True, random_initalization=False)
    elif (ui_1 != 1) and (ui_2 == 1):
        epoch = 100
        model = perceptron(epoch=epoch, lr=lr)
        if if_batch == 1:
            model.bl(X, y_lable, shuffle=False, random_initalization=True)
        else:
            model.ol(X, y_lable, shuffle=False, random_initalization=True)
    elif (ui_1 != 1) and (ui_2 != 1):
        model = perceptron(epoch=epoch, lr=lr)
        if if_batch == 1:
            model.bl(X, y_lable)
        else:
            model.ol(X, y_lable)
    else:
        epoch = 100
        model = perceptron(epoch=epoch, lr=lr)
        if if_batch == 1:
            model.bl(X, y_lable, shuffle=True, random_initalization=True)
        else:
            model.ol(X, y_lable, shuffle=True, random_initalization=True)
    w = model.w
    w = np.array(w)
    w = w.T
    weights = []
    for i in range(len(w)):
        weights.append(float(w[i]))
    w = np.array(weights)
    # -----------------------------------------------------------#

    # --- Hyperplane plotting --- #
    xx = np.array(X[:, 1])
    yy = np.array(X[:, 2])
    x = []
    y = []
    for i in range(len(xx)):
        x.append(float(xx[i]))
        y.append(float(yy[i]))
    m, n = np.meshgrid(np.linspace(np.min(x), np.max(x), num=100),
                       np.linspace(np.min(y), np.max(y), num=100))
    z = (-(w[2]) * n - (2 * w[1]) * m - w[0]) / (w[3])
    for i in range(3):
        idx = np.where(Y == i)
        axes.scatter(X[idx, 1], X[idx, 2], X[idx, 3], cmap=plt.cm.Set1,
                     edgecolor='k', s=40, label=y_classes[i])
    axes.plot_surface(m, n, z, rstride=1, cstride=1,
                      cmap='viridis', edgecolor='none')
    axes.legend()
    if if_batch == 1:
        axes.set_title("Claasification using Batch learning, One vs all")
    else:
        axes.set_title("Claasification using Online learning, One vs all")
    axes.set_xlabel(y_classes[0])
    axes.w_xaxis.set_ticklabels([])
    axes.set_ylabel(y_classes[1])
    axes.w_yaxis.set_ticklabels([])
    axes.set_zlabel(y_classes[2])
    axes.w_zaxis.set_ticklabels([])
    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    # --- To be consistent we will iterate for same no. of times in each cases --- #
    epoch = 10
    model = perceptron(epoch=epoch, lr=1)
    model.ol(X, y_lable)
    axes.plot(range(1, epoch + 1), model.error, marker='*', color='b')
    plt.show()

# -----------------------------------------------------------#

top = Tk()

top.geometry("200x200")
messagebox.showinfo('Message 1', "You might like to rotate the figure for better view")
messagebox.showinfo('Message 2',
                    "As in two cases there are no classifying hyperplance so they are not linearly sperable")

top.mainloop()
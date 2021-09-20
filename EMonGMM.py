from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
data = np.loadtxt("gmm_data.txt", delimiter=" ", dtype=float)
data = np.array(data)
plt.scatter(data[:, 0], data[:, 1])

class em_gmm:
    def __init__(self, x,k):
        self.pnk=[]
        self.k=k
        self.a, self.b = x.shape
        #generating k random integers to initialize [580,618,695,490,889]
        r = np.random.randint(low=0, high=self.a, size=self.k)
        #these are the idex for which program works best 
        r=[580,618,695,490,889]
        self.m=[]
        for i in r:
            #mu denotes mean and c denotes variance 
            temp=x[i,:]
            self.m.append(temp)#choosing for temp index values as mean
        self.c=[]
        for i in range(self.k):
            temp=self.cov(x, self.m[i])#finding covariance for initialized mean
            self.c.append(temp)
    def cov(self,x,mu): #function to find covariance 
        c=[] 
        for i in range(self.b):
            temp=[]
            s=0
            for j in range(self.b):
                for k in range(self.a):
                    cx=x[k][i]-mu[i]#defining cx and cy as x-mean and y-mean
                    cy=x[k][j]-mu[j]
                    s=s+cx*cy    #summing all multiplication of cx and cy
                temp.append(s/self.a)
            c.append(temp)
        return c
    def prob(self,x):
        #l stands for likelihood
        l = np.zeros([self.a, self.k])# taking an 0 matrix to intialize
        self.w = np.full(shape=self.k, fill_value=1 /self.k)
        for i in range(self.k):
            dist= multivariate_normal(mean=self.m[i],cov=self.c[i])# as mixture is gausian finding distribution for intialized mean and variance to start and then iteretivey updating
            l[:,i] = dist.pdf(x)
        self.pnk=[]
        for i in range(self.a):
            temp = []
            denom=np.sum(np.multiply(self.w, l[i, :]))
            for j in range(self.k):
                temp.append(self.w[j]*l[i, j]/denom)#finidng pnk by formula 
            self.pnk.append(temp)
        self.pnk = np.array(self.pnk)
        return self.pnk    
    def expect(self, x):
        self.prob(x) 
    def em(self,x):# em denotes expectation maimization 
        #this function will ittretively update covariance and mean by eqtns 
        Nk=[]
        self.Wk=[]
        temp=0
        self.m=[]
        self.c=[]
        for i in range(self.k):
            temp_ck = np.zeros([self.b, self.b])
            temp=np.sum(self.pnk[:,i])
            Nk.append(temp)
            self.Wk.append(temp/self.a)
            temp_=[0,0]
            
            for j in range(self.a):
                temp_m=(np.multiply(self.pnk[j, i], x[j, :])/temp)
                temp_=temp_+temp_m
            self.m.append(temp_)
            ck_1 = [(np.outer((x[j, :]-self.m[i]), (x[j, :]-self.m[i]).T)) for j in range(self.a)]
            ck_2=[np.multiply(ck_1[j],self.pnk[j,i]) for j in range(self.a)]
            for j in range(self.a):
                temp_ck=temp_ck+ck_2[j]
            temp_ck=temp_ck/temp
            self.c.append(temp_ck)
    def learn(self, x,iter):
        for _ in range(iter):
            self.expect(x)
            self.em(x)
    def pred(self, x):
        lab=self.prob(x)
        return np.argmax(lab,axis=1)# finding those arguments where probability is maximum
k=5     
model=em_gmm(data,k)
model.learn(data,50) # works well for those indexes and iteration(I have taken 50)>16 
labels=model.pred(data) 
def plot_ellipse(position, covariance, ax=None,**kwargs):# function to plot the ellipse
    ax = ax or plt.gca()
    # PCA
    c=covariance
    if c.shape == (2, 2):
        U, s, V = np.linalg.svd(c)
        ang = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        wi, hi = 2 * np.sqrt(s)
    else:
        ang = 0
        wi, hi = 2 * np.sqrt(c)
    # Draw the Ellipse
    for i in range(1, 4):
        ellipse=Ellipse(position, i * wi, i * hi, ang,**kwargs)
        ax.add_artist(ellipse)
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
w = 0.4/ model.w.max()
for i in range(k):
    plot_ellipse(model.m[i],model.c[i],alpha= model.w[i] * w)
plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap='cividis')
plt.show()

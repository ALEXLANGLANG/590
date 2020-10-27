import numpy as np
import matplotlib.pyplot as plt

#General plot for b and c
def plot_(list_L, epochs, list_W, label = ""):
    plt.figure(figsize = (12,4))
    plt.subplot(1,2,1)
    plt.title("epochs vs. Log(L)" + label)
    plt.plot(range(epochs),np.log(list_L), '-')
    plt.xlabel("epochs")
    plt.ylabel("log(L)")
    plt.subplot(1,2,2)
    plt.title("epochs vs. W"+ label)
    list_W = np.array(list_W)
    for i in range(5):
        plt.plot(range(epochs),list_W[:,i], '-', label = 'w_'+str(i+1))
    plt.xlabel("epochs")
    plt.ylabel("value of element in W")
    plt.legend()
    plt.show()
    
def do_b(X,Y,W0,u,epochs):
    W = W0
    list_W = []
    list_L = []
    for i in range(epochs):
        W = W - X.T @ (2*u*(X@W - Y))

        L = sum((X@W - Y)**2)
        list_L.append(L)
        list_W.append(W)

    plot_(list_L, epochs, list_W, label = "")



def do_c(X,Y,W0,u,epochs):
    W = W0
    list_W = []
    list_L = []
    for i in range(epochs):
        W = W - X.T @ (2*u*(X@W - Y))
        sort_W = np.sort(np.abs(W))[:3]
        for i in range(5):
            if abs(W[i]) in sort_W:
                W[i] = 0
        L = sum((X@W - Y)**2)
        list_L.append(L)
        list_W.append(W)
    
    plot_(list_L, epochs, list_W, label = " for Iterative pruning")
    
    
def do_d(X,Y,W0,u,epochs):
    for lambda_ in [0.2, 0.5, 1.0, 2.0]:
        W = W0
        list_L = []
        list_W = []
        epochs = 200
        for i in range(epochs):
            W = W - u*(2* X.T@((X@W - Y))+ lambda_* np.sign(W))
            L = sum((X@W - Y)**2)
            list_L.append(L)
            list_W.append(W)
        plot_(list_L, epochs, list_W, label = " for L1 when lambda = " + str(lambda_) )
        
        
def proxL1(lambda_,W):
    proxl1 = []
    for theta in W:
        if theta > lambda_:
            proxl1.append(theta -lambda_)
        elif theta < -lambda_:
            proxl1.append(theta + lambda_)
        else:
            proxl1.append(0)
    return np.array(proxl1) 


def do_e(X,Y,W0,u,epochs):
    for lambda_ in [0.2, 0.5, 1.0, 2.0]:
        W = W0
        list_L = []
        list_W = []
        for i in range(epochs):
            W = W - u*(2*X.T@(X@W - Y))
            W = proxL1(u*lambda_,W)
            L = sum((X@W - Y)**2)
            list_L.append(L)
            list_W.append(W)
        plot_(list_L, epochs, list_W, label = " for Proximal L1 when threshold = " + str(lambda_*u) )
        
        
def do_f(X,Y,W0,u,epochs):
    for lambda_ in [1.0, 2.0, 5.0, 10.0]:
        W = W0
        list_L = []
        list_W = []
        epochs = 200
        for i in range(epochs):
            W = W - u*(2*X.T@(X@W - Y)) # Get gradient
           
            index_min = np.argsort(abs(W))[:3] # Get the index of three smallest element in weights
            W[index_min] = proxL1(u*lambda_, W[index_min]) # Do proximal only on three smallest element in weights
            L = sum((X@W - Y)**2)
            list_L.append(L)
            list_W.append(W)
        plot_(list_L, epochs, list_W, label = " for Trimmed L1 when threshold = " + str(lambda_*u) )




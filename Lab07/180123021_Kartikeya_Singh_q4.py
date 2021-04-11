import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from math import log, sqrt, erf, exp

def d(x, tau, sig, K, r):
    d_plus = (1/(sig*sqrt(tau)))*(log(x/K) + tau*(r + (sig*sig)/2))
    d_minus = (1/(sig*sqrt(tau)))*(log(x/K) + tau*(r - (sig*sig)/2))

    return d_plus, d_minus

def N(x):
	return 0.5*(1 + erf(x/sqrt(2)))

def C(t, x, T, sig, K, r):
    if x == 0:
        return 0
    if t == T:
        return max(x - K, 0)

    tau = T-t
    d_plus, d_minus = d(x, tau, sig, K, r)
    
    price = x*N(d_plus) - K*exp(-r*tau)*N(d_minus)

    return price

def P(t, x, T, sig, K, r):
    call = C(t,x,T,sig,K,r)
    put = call + K*exp(-r*(T-t)) - x

    return put

def OptionPrices(S,K,T,t,r,sigma):
    if t==T:
        return max(S-K,0),max(K-S,0)

    d1 = 1/(sigma * sqrt(T-t)) * (log(S/K) + (r + (sigma**2)/2)*(T-t))
    d2 = d1 - sigma*sqrt(T-t)
    PV = K*exp(-r*(T-t))

    Call = N(d1)*S - N(d2)*PV
    Put = -N(-d1)*S + N(-d2)*PV

    return Call,Put

def plot2D(X, Call, Put, param):
    
    plt.plot(X,Call, label = "Call Option")
    plt.plot(X,Put,label = "Put Option")
    plt.xlabel(param)
    plt.ylabel("Option Price")
    plt.title("Varying %s"%param)
    plt.legend(loc = 'best')
    plt.savefig(f"q4_{param}")
    plt.clf()

def plot3D(X, Y, Call, Put, param1, param2):
    fig = plt.figure(figsize=(9,9))
    fig.suptitle("Varying %s and %s "%(param1,param2))

    ax = fig.add_subplot(2,1,1,projection = '3d')
    ax.plot_surface(np.array(X), np.array(Y), np.array(Call), cmap='viridis', edgecolor='none')
    ax.set_title("Call Option Price")
    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    ax.set_zlabel("Option Price")

    ax = fig.add_subplot(2,1,2,projection = '3d')
    ax.plot_surface(np.array(X), np.array(Y), np.array(Put), cmap='viridis', edgecolor='none')
    ax.set_title("Put Option Price")
    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    ax.set_zlabel("Option Price")
    
    plt.savefig(f"q4_{param1}_{param2}")
    plt.clf()

def analyzeSensitivity(T, K, r, sig, param):
    t = 0.1
    x = 1

    T_range = np.linspace(0.1, 1.9, 1000)
    K_range = np.linspace(0.1, 1.9, 1000)
    r_range = np.linspace(0.01, 0.51, 1000)
    sig_range = np.linspace(0.3, 0.9, 1000)

    X = []
    Call = []
    Put =  []

    if param == "T":
        for T1 in T_range:
            call = C(t, x, T1, sig, K, r)
            put = P(t, x, T1, sig, K, r)
            X.append(T1)
            Call.append(call)
            Put.append(put)
    elif param == "K":
        for K1 in K_range:
            call = C(t, x, T, sig, K1, r)
            put = P(t, x, T, sig, K1, r)
            X.append(K1)
            Call.append(call)
            Put.append(put)
    elif param == "r":
        for r1 in r_range:
            call = C(t, x, T, sig, K, r1)
            put = P(t, x, T, sig, K, r1)
            X.append(r1)
            Call.append(call)
            Put.append(put)
    elif param == "sig":
        for sig1 in sig_range:
            call = C(t, x, T, sig1, K, r)
            put = P(t, x, T, sig1, K, r)
            X.append(sig1)
            Call.append(call)
            Put.append(put)

    plot2D(X, Call, Put, param)


def analyzeSensitivity2(T, K, r, sig, param1, param2):
    t = 0.1
    x = 1

    T_range = np.linspace(0.1, 1.9, 100)
    K_range = np.linspace(0.1, 1.9, 100)
    r_range = np.linspace(0.01, 0.21, 100)
    sig_range = np.linspace(0.3, 0.9, 100)

    X = []
    Y = []
    Call = []
    Put = []

    if param1 == "T" and param2 == "K":
        for T1 in T_range:
            X_temp = []
            Y_temp = []
            Call_temp = []
            Put_temp = []
            for K1 in K_range:
                call = C(t, x, T1, sig, K1, r)
                put = P(t, x, T1, sig, K1, r)
                X_temp.append(T1)
                Y_temp.append(K1)
                Call_temp.append(call)
                Put_temp.append(put)
            X.append(X_temp)
            Y.append(Y_temp)
            Call.append(Call_temp)
            Put.append(Put_temp)    
    elif param1 == "T" and param2 == "r":
        for T1 in T_range:
            X_temp = []
            Y_temp = []
            Call_temp = []
            Put_temp = []
            for r1 in r_range:
                call = C(t, x, T1, sig, K, r1)
                put = P(t, x, T1, sig, K, r1)
                X_temp.append(T1)
                Y_temp.append(r1)
                Call_temp.append(call)
                Put_temp.append(put)
            X.append(X_temp)
            Y.append(Y_temp)
            Call.append(Call_temp)
            Put.append(Put_temp) 
    elif param1 == "T" and param2 == "sig":
        for T1 in T_range:
            X_temp = []
            Y_temp = []
            Call_temp = []
            Put_temp = []
            for sig1 in sig_range:
                call = C(t, x, T1, sig1, K, r)
                put = P(t, x, T1, sig1, K, r)
                X_temp.append(T1)
                Y_temp.append(sig1)
                Call_temp.append(call)
                Put_temp.append(put)
            X.append(X_temp)
            Y.append(Y_temp)
            Call.append(Call_temp)
            Put.append(Put_temp) 
    elif param1 == "K" and param2 == "r":
        for K1 in K_range:
            X_temp = []
            Y_temp = []
            Call_temp = []
            Put_temp = []
            for r1 in r_range:
                call = C(t, x, T, sig, K1, r1)
                put = P(t, x, T, sig, K1, r1)
                X_temp.append(K1)
                Y_temp.append(r1)
                Call_temp.append(call)
                Put_temp.append(put)
            X.append(X_temp)
            Y.append(Y_temp)
            Call.append(Call_temp)
            Put.append(Put_temp) 
    elif param1 == "K" and param2 == "sig":
        for K1 in K_range:
            X_temp = []
            Y_temp = []
            Call_temp = []
            Put_temp = []
            for sig1 in sig_range:
                call = C(t, x, T, sig1, K1, r)
                put = P(t, x, T, sig1, K1, r)
                X_temp.append(K1)
                Y_temp.append(sig1)
                Call_temp.append(call)
                Put_temp.append(put)
            X.append(X_temp)
            Y.append(Y_temp)
            Call.append(Call_temp)
            Put.append(Put_temp) 
    elif param1 == "r" and param2 == "sig":
        for r1 in r_range:
            X_temp = []
            Y_temp = []
            Call_temp = []
            Put_temp = []
            for sig1 in sig_range:
                call = C(t, x, T, sig1, K, r1)
                put = P(t, x, T, sig1, K, r1)
                X_temp.append(r1)
                Y_temp.append(sig1)
                Call_temp.append(call)
                Put_temp.append(put)
            X.append(X_temp)
            Y.append(Y_temp)
            Call.append(Call_temp)
            Put.append(Put_temp)

    plot3D(X, Y, Call, Put, param1, param2) 


def main():
    T = 1
    K = 1
    r = 0.05
    sig = 0.6

    params = ["T", "K", "r", "sig"]

    for param in params:
        analyzeSensitivity(T,K,r,sig,param)

    for i in range(len(params)-1):
        for j in range(i+1,len(params)):
            analyzeSensitivity2(T,K,r,sig,params[i],params[j])
                
if __name__ == '__main__':
    main()
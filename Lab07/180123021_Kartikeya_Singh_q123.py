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

def plot3D(X, Y, Z, option_type):
    ax = plt.axes(projection='3d')
    ax.plot_surface(np.array(X), np.array(Y), np.array(Z), cmap='viridis', edgecolor='none')
    ax.set_title("%s Option Price as a function of (t, x)"%option_type)
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_zlabel("%s Option Price"%option_type)
    plt.show()
    plt.clf()

def plot3Dscatter(X, Y, Z, option_type):
    ax = plt.axes(projection='3d')
    ax.scatter(np.array(X), np.array(Y), np.array(Z), cmap='viridis', edgecolor='none')
    ax.set_title("%s Option Price as a function of (t, x)"%option_type)
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_zlabel("%s Option Price"%option_type)
    plt.show()
    plt.clf()

def q2(T, K, r, sig, times):

    X = np.linspace(0, 2, 1000)

    # Call option 2-D
    for t in times:
        prices = [C(t,x,T,sig,K,r) for x in X]
        plt.plot(X, prices, label = "t = %s"%t)

    plt.xlabel("x")
    plt.ylabel("Call Option Price (C(t,x))")
    plt.title("Call Option Price (C(t,x)) vs Price of underlying asset (x)")
    plt.legend(loc = "best")
    plt.show()
    plt.clf()

    # Put option 2-D
    for t in times:
        prices = [P(t,x,T,sig,K,r) for x in X]
        plt.plot(X, prices, label = "t = %s"%t)

    plt.xlabel("x")
    plt.ylabel("Put Option Price (P(t,x))")
    plt.title("Put Option Price (P(t,x)) vs Price of underlying asset (x)")
    plt.legend(loc = "best")
    plt.show()
    plt.clf()

    # 3-D graphs
    X_vals = np.linspace(0, 2, 1000)
    
    T_pts = []
    X_pts = []

    call_prices = []
    put_prices = []

    for t in times:
        call_temp = []
        put_temp = []
        t_temp = []
        x_temp = []
        for x in X_vals:
            call = C(t,x,T,sig, K,r)
            put = P(t,x,T,sig,K,r)
            t_temp.append(t)
            x_temp.append(x)
            call_temp.append(call)
            put_temp.append(put)
        T_pts.append(t_temp)
        X_pts.append(x_temp)
        call_prices.append(call_temp)
        put_prices.append(put_temp)

    plot3Dscatter(T_pts, X_pts, call_prices, "Call")
    plot3Dscatter(T_pts, X_pts, put_prices, "Put")

def q3(T, K, r, sig):
    T_vals = np.linspace(0, T, 1000)
    X_vals = np.linspace(0, 2, 1000)
    
    T_pts = []
    X_pts = []

    call_prices = []
    put_prices = []

    for t in T_vals:
        call_temp = []
        put_temp = []
        t_temp = []
        x_temp = []
        for x in X_vals:
            call = C(t,x,T,sig, K,r)
            put = P(t,x,T,sig,K,r)
            t_temp.append(t)
            x_temp.append(x)
            call_temp.append(call)
            put_temp.append(put)
        T_pts.append(t_temp)
        X_pts.append(x_temp)
        call_prices.append(call_temp)
        put_prices.append(put_temp)

    plot3D(T_pts, X_pts, call_prices, "Call")
    plot3D(T_pts, X_pts, put_prices, "Put")    

def main():
    T = 1
    K = 1
    r = 0.05
    sig = 0.6

    times = [0, 0.2, 0.4, 0.6, 0.8, 1]

    q2(T, K, r, sig, times)
    q3(T, K, r, sig)
                
if __name__ == '__main__':
    main()
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm

def calcJump(setNo,sig,dt,r):
    if setNo == 1:
        u = np.exp(sig*(dt**0.5))
        d = np.exp(-sig*(dt**0.5))
    else:
        u = np.exp(sig*(dt**0.5) + (r - 0.5*(sig**2))*dt)
        d = np.exp(-sig*(dt**0.5) + (r - 0.5*(sig**2))*dt)
    return u,d

def calcOptionPrice(setNo,M,T,S0,r,K,sig):
    size = M+1
    dt = T/M

    u,d = calcJump(setNo,sig,dt,r)

    p = (np.exp(r*dt)-d)/(u-d) 

    # call[t][i] represents the value of a call option
    # at time 't' if the underlying stock goes down 
    # 'i' times and goes up 't-i' times.    
    call = np.zeros([size,size])
    put = np.zeros([size,size])

    for i in range(size):
        Sn = S0 * (d**i) * (u**(M-i))
        put[M][i] = max(K-Sn,0)
        call[M][i] = max(Sn-K,0)
    
    for t in range(M-1,-1,-1):
        for i in range(t+1):
            put[t][i] = np.exp(-r*dt) * (p*put[t+1][i] + (1-p)*put[t+1][i+1])
            call[t][i] = np.exp(-r*dt) * (p*call[t+1][i] + (1-p)*call[t+1][i+1])

    return call[0][0], put[0][0]

def plot2D(X,Put1,Call1,Put2,Call2,title,param,figname):
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.suptitle(title)
    ax1.set_xlabel(param)
    ax1.set_ylabel("Option Price")
    ax1.set_title("Set - 1")
    ax1.plot(X,Put1,label = "Put Option")
    ax1.plot(X,Call1,label = "Call Option")
    ax1.legend(loc = 'upper right')
    ax2.set_xlabel(param)
    ax2.set_ylabel("Option Price")
    ax2.set_title("Set - 2")
    ax2.plot(X,Put2,label = "Put Option")
    ax2.plot(X,Call2,label = "Call Option")
    ax2.legend(loc = 'upper right')
    
    plt.tight_layout()
    plt.savefig(figname)

def plot3D(X,Y,Put1,Call1,Put2,Call2,param1,param2,figname):
    fig = plt.figure(figsize=(9,9))
    fig.suptitle("Varying %s and %s "%(param1,param2))

    ax = fig.add_subplot(2,2,1,projection = '3d')
    ax.set_title("Put Option (Set-1)")
    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    ax.set_zlabel("Option Price")
    ax.plot_surface(np.array(X),np.array(Y),np.array(Put1),cmap = cm.coolwarm)

    ax = fig.add_subplot(2,2,2,projection = '3d')
    ax.set_title("Put Option (Set-2)")
    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    ax.set_zlabel("Option Price")
    ax.plot_surface(np.array(X),np.array(Y),np.array(Put2),cmap = cm.coolwarm)

    ax = fig.add_subplot(2,2,3,projection = '3d')
    ax.set_title("Call Option (Set-1)")
    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    ax.set_zlabel("Option Price")
    ax.plot_surface(np.array(X),np.array(Y),np.array(Call1),cmap = cm.coolwarm)
    
    ax = fig.add_subplot(2,2,4,projection = '3d')
    ax.set_title("Call Option (Set-2)")
    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    ax.set_zlabel("Option Price")
    ax.plot_surface(np.array(X),np.array(Y),np.array(Call2),cmap = cm.coolwarm)

    plt.savefig(figname)

def analyzeSensitivity(S0,K,T,M,r,sig,param):
    if param == "M":
        k = [95,100,105]
        for k1 in k:
            X = []
            Put1 = []
            Call1 = []
            Put2 = []
            Call2 = []
            for m in range(50,151):
                call1,put1 = calcOptionPrice(1,m,T,S0,r,k1,sig)
                call2,put2 = calcOptionPrice(2,m,T,S0,r,k1,sig)
                Put1.append(put1)
                Call1.append(call1)
                Put2.append(put2)
                Call2.append(call2)
                X.append(m)
            title = "Varying M with K = %d"%k1
            figname = "Q1_%s_%d.png"%(param,k1)
            plot2D(X,Put1,Call1,Put2,Call2,title,param,figname)
    else:
        X = []
        Put1 = []
        Call1 = []
        Put2 = []
        Call2 = []
        if param == "S0":
            for s0 in range(50,151):
                call1,put1 = calcOptionPrice(1,M,T,s0,r,K,sig)
                call2,put2 = calcOptionPrice(2,M,T,s0,r,K,sig)
                Put1.append(put1)
                Call1.append(call1)
                Put2.append(put2)
                Call2.append(call2)
                X.append(s0)
        elif param == "K":
            for k in range(50,151):
                call1,put1 = calcOptionPrice(1,M,T,S0,r,k,sig)
                call2,put2 = calcOptionPrice(2,M,T,S0,r,k,sig)
                Put1.append(put1)
                Call1.append(call1)
                Put2.append(put2)
                Call2.append(call2)
                X.append(k)
        elif param == "r":
            for R in range(500,1005,5):
                r1 = R/10000
                call1,put1 = calcOptionPrice(1,M,T,S0,r1,K,sig)
                call2,put2 = calcOptionPrice(2,M,T,S0,r1,K,sig)
                Put1.append(put1)
                Call1.append(call1)
                Put2.append(put2)
                Call2.append(call2)
                X.append(r1)
        elif param == "sigma":
            for Sigma in range(1500,2500,10):
                sigma = Sigma/10000
                call1,put1 = calcOptionPrice(1,M,T,S0,r,K,sigma)
                call2,put2 = calcOptionPrice(2,M,T,S0,r,K,sigma)
                Put1.append(put1)
                Call1.append(call1)
                Put2.append(put2)
                Call2.append(call2)
                X.append(sigma)
        title = "Varying %s"%param
        figname = "Q1_%s.png"%param
        plot2D(X,Put1,Call1,Put2,Call2,title,param,figname)

def analyzeSensitivity2(S0,K,T,M,r,sig,param1,param2):
    X = []
    Y = []
    Put1 = []
    Call1 = []
    Put2 = []
    Call2 = []
    if param1 == "S0" and param2 == "K":
        for s0 in range(40,162,5):
            x_temp = []
            y_temp = []
            put1_temp = []
            call1_temp = []
            put2_temp = []
            call2_temp = []
            for k in range(40,162,5):
                x_temp.append(s0)
                y_temp.append(k)
                call1,put1 = calcOptionPrice(1,M,T,s0,r,k,sig)
                call2,put2 = calcOptionPrice(2,M,T,s0,r,k,sig)
                call1_temp.append(call1)
                put1_temp.append(put1)
                call2_temp.append(call2)
                put2_temp.append(put2)
            X.append(x_temp)
            Y.append(y_temp)
            Put1.append(put1_temp)
            Call1.append(call1_temp)
            Put2.append(put2_temp)
            Call2.append(call2_temp)
    elif param1 == "S0" and param2 == "r":
        for s0 in range(40,162,5):
            x_temp = []
            y_temp = []
            put1_temp = []
            call1_temp = []
            put2_temp = []
            call2_temp = []
            for R in range(500,1000,10):
                r1 = R/10000
                x_temp.append(s0)
                y_temp.append(r1)
                call1,put1 = calcOptionPrice(1,M,T,s0,r1,K,sig)
                call2,put2 = calcOptionPrice(2,M,T,s0,r1,K,sig)
                call1_temp.append(call1)
                put1_temp.append(put1)
                call2_temp.append(call2)
                put2_temp.append(put2)
            X.append(x_temp)
            Y.append(y_temp)
            Put1.append(put1_temp)
            Call1.append(call1_temp)
            Put2.append(put2_temp)
            Call2.append(call2_temp)

    elif param1 == "S0" and param2 == "sigma":
        for s0 in range(40,162,5):
            x_temp = []
            y_temp = []
            put1_temp = []
            call1_temp = []
            put2_temp = []
            call2_temp = []
            for Sigma in range(1750,2250,10):
                sigma = Sigma/10000
                x_temp.append(s0)
                y_temp.append(sigma)
                call1,put1 = calcOptionPrice(1,M,T,s0,r,K,sigma)
                call2,put2 = calcOptionPrice(2,M,T,s0,r,K,sigma)
                call1_temp.append(call1)
                put1_temp.append(put1)
                call2_temp.append(call2)
                put2_temp.append(put2)
            X.append(x_temp)
            Y.append(y_temp)
            Put1.append(put1_temp)
            Call1.append(call1_temp)
            Put2.append(put2_temp)
            Call2.append(call2_temp)
    elif param1 == "S0" and param2 == "M":
        for s0 in range(40,162,5):
            x_temp = []
            y_temp = []
            put1_temp = []
            call1_temp = []
            put2_temp = []
            call2_temp = []
            for m in range(40,162,5):
                x_temp.append(s0)
                y_temp.append(m)
                call1,put1 = calcOptionPrice(1,m,T,s0,r,K,sig)
                call2,put2 = calcOptionPrice(2,m,T,s0,r,K,sig)
                call1_temp.append(call1)
                put1_temp.append(put1)
                call2_temp.append(call2)
                put2_temp.append(put2)
            X.append(x_temp)
            Y.append(y_temp)
            Put1.append(put1_temp)
            Call1.append(call1_temp)
            Put2.append(put2_temp)
            Call2.append(call2_temp)
    elif param1 == "K" and param2 == "r":
        for k in range(40,162,5):
            x_temp = []
            y_temp = []
            put1_temp = []
            call1_temp = []
            put2_temp = []
            call2_temp = []
            for R in range(500,1000,10):
                r1 = R/10000
                x_temp.append(k)
                y_temp.append(r1)
                call1,put1 = calcOptionPrice(1,M,T,S0,r1,k,sig)
                call2,put2 = calcOptionPrice(2,M,T,S0,r1,k,sig)
                call1_temp.append(call1)
                put1_temp.append(put1)
                call2_temp.append(call2)
                put2_temp.append(put2)
            X.append(x_temp)
            Y.append(y_temp)
            Put1.append(put1_temp)
            Call1.append(call1_temp)
            Put2.append(put2_temp)
            Call2.append(call2_temp)
    elif param1 == "K" and param2 == "sigma":
        for k in range(40,162,5):
            x_temp = []
            y_temp = []
            put1_temp = []
            call1_temp = []
            put2_temp = []
            call2_temp = []
            for Sigma in range(1750,2250,10):
                sigma = Sigma/10000
                x_temp.append(k)
                y_temp.append(sigma)
                call1,put1 = calcOptionPrice(1,M,T,S0,r,k,sigma)
                call2,put2 = calcOptionPrice(2,M,T,S0,r,k,sigma)
                call1_temp.append(call1)
                put1_temp.append(put1)
                call2_temp.append(call2)
                put2_temp.append(put2)
            X.append(x_temp)
            Y.append(y_temp)
            Put1.append(put1_temp)
            Call1.append(call1_temp)
            Put2.append(put2_temp)
            Call2.append(call2_temp)
    elif param1 == "K" and param2 == "M":
        for k in range(40,162,5):
            x_temp = []
            y_temp = []
            put1_temp = []
            call1_temp = []
            put2_temp = []
            call2_temp = []
            for m in range(40,162,5):
                x_temp.append(k)
                y_temp.append(m)
                call1,put1 = calcOptionPrice(1,m,T,S0,r,k,sig)
                call2,put2 = calcOptionPrice(2,m,T,S0,r,k,sig)
                call1_temp.append(call1)
                put1_temp.append(put1)
                call2_temp.append(call2)
                put2_temp.append(put2)
            X.append(x_temp)
            Y.append(y_temp)
            Put1.append(put1_temp)
            Call1.append(call1_temp)
            Put2.append(put2_temp)
            Call2.append(call2_temp)
    elif param1 == "r" and param2 == "sigma":
        for R in range(500,1000,10):
            r1 = R/10000
            x_temp = []
            y_temp = []
            put1_temp = []
            call1_temp = []
            put2_temp = []
            call2_temp = []
            for Sigma in range(1750,2250,10):
                sigma = Sigma/10000
                x_temp.append(r1)
                y_temp.append(sigma)
                call1,put1 = calcOptionPrice(1,M,T,S0,r1,K,sigma)
                call2,put2 = calcOptionPrice(2,M,T,S0,r1,K,sigma)
                call1_temp.append(call1)
                put1_temp.append(put1)
                call2_temp.append(call2)
                put2_temp.append(put2)
            X.append(x_temp)
            Y.append(y_temp)
            Put1.append(put1_temp)
            Call1.append(call1_temp)
            Put2.append(put2_temp)
            Call2.append(call2_temp)
    elif param1 == "r" and param2 == "M":
        for R in range(500,1000,10):
            r1 = R/10000
            x_temp = []
            y_temp = []
            put1_temp = []
            call1_temp = []
            put2_temp = []
            call2_temp = []
            for m in range(40,162,5):
                x_temp.append(r1)
                y_temp.append(m)
                call1,put1 = calcOptionPrice(1,m,T,S0,r1,K,sig)
                call2,put2 = calcOptionPrice(2,m,T,S0,r1,K,sig)
                call1_temp.append(call1)
                put1_temp.append(put1)
                call2_temp.append(call2)
                put2_temp.append(put2)
            X.append(x_temp)
            Y.append(y_temp)
            Put1.append(put1_temp)
            Call1.append(call1_temp)
            Put2.append(put2_temp)
            Call2.append(call2_temp)
    elif param1 == "sigma" and param2 == "M":
        for Sigma in range(1750,2250,10):
            sigma = Sigma/10000
            x_temp = []
            y_temp = []
            put1_temp = []
            call1_temp = []
            put2_temp = []
            call2_temp = []
            for m in range(40,162,5):
                x_temp.append(sigma)
                y_temp.append(m)
                call1,put1 = calcOptionPrice(1,m,T,S0,r,K,sigma)
                call2,put2 = calcOptionPrice(2,m,T,S0,r,K,sigma)
                call1_temp.append(call1)
                put1_temp.append(put1)
                call2_temp.append(call2)
                put2_temp.append(put2)
            X.append(x_temp)
            Y.append(y_temp)
            Put1.append(put1_temp)
            Call1.append(call1_temp)
            Put2.append(put2_temp)
            Call2.append(call2_temp)
    figname = "Q1_%s_%s.png"%(param1,param2)
    plot3D(X,Y,Put1,Call1,Put2,Call2,param1,param2,figname)

def main():
    S0 = 100.0
    K = 100.0
    T = 1
    M = 100
    r = 0.08
    sig = 0.2

    call1, put1 = calcOptionPrice(1,M,T,S0,r,K,sig)
    call2, put2 = calcOptionPrice(2,M,T,S0,r,K,sig)

    print("Call Option price for Set-1 = %.6f"%call1)
    print("Put Option price for Set-1 = %.6f"%put1)
    print("Call Option price for Set-2 = %.6f"%call2)
    print("Put Option price for Set-2 = %.6f"%put2)
    
    params = ["S0","K","r","sigma","M"]

    for param in params:
        analyzeSensitivity(S0,K,T,M,r,sig,param)

    for i in range(len(params)-1):
        for j in range(i+1,len(params)):
            analyzeSensitivity2(S0,K,T,M,r,sig,params[i],params[j])          

if __name__ == '__main__':
    main()

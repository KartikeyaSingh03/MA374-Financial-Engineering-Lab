import numpy as np
import matplotlib.pyplot as plt

# calculate u and d
def calcJump(sig,dt,r):
    u = np.exp(sig*(dt**0.5) + (r - 0.5*(sig**2))*dt)
    d = np.exp(-sig*(dt**0.5) + (r - 0.5*(sig**2))*dt)
    return u,d

def calcOptionPrice(u,d,p,M,dt,S0,r,K):
    size = M+1

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

def plotGraph(opt,step,x,y):
    plt.plot(x,y,linewidth=0.5)
    plt.title("%s Option Price \n with step = %d" % (opt, step))
    plt.xlabel("M (Number of steps)")
    plt.ylabel("European %s Option Price"%opt)
    plt.savefig("%s_%s"%(opt,step))
    plt.clf()

def compute(step):
    S0 = 100.0
    K = 105.0
    T = 5
    r = 0.05
    sig = 0.3

    M = []
    for i in range(1,402,step):
        M.append(i)

    call_prices = []
    put_prices = []

    for m in M:
        dt = T/m   
        u,d = calcJump(sig,dt,r)
        p = (np.exp(r*dt)-d)/(u-d) 

        if d<np.exp(r*dt) and np.exp(r*dt)<u:
            callPrice, putPrice = calcOptionPrice(u,d,p,m,dt,S0,r,K)
            call_prices.append(callPrice)
            put_prices.append(putPrice)
        else:
            print("There is an Arbitrage Opportunity for M = %d"%m)
            return

    # print("The Call Prices for step = %d are:"%step)
    # print(call_prices)
    # print("The Put Prices for step = %d are:"%step)
    # print(put_prices)

    plotGraph("Call",step,M,call_prices)
    plotGraph("Put",step,M,put_prices)

def main():
   compute(1)
   compute(5)
        
if __name__ == "__main__":
    main()
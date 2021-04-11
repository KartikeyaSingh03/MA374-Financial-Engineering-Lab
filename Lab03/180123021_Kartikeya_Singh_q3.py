import numpy as np
import time

def calcJump(sig,dt,r):
    u = np.exp(sig*(dt**0.5) + (r - 0.5*(sig**2))*dt)
    d = np.exp(-sig*(dt**0.5) + (r - 0.5*(sig**2))*dt)
    return u,d

def calcOptionPrice(M,T,S0,r,K,sig):
    size = M+1
    dt = T/M

    u,d = calcJump(sig,dt,r)

    p = (np.exp(r*dt)-d)/(u-d) 

    # call[t][i] represents the value of a call option
    # at time 't' if the underlying stock goes down 
    # 'i' times and goes up 't-i' times.    
    call = np.zeros([size,size])

    for i in range(size):
        Sn = S0 * (d**i) * (u**(M-i))
        call[M][i] = max(Sn-K,0)
    
    for t in range(M-1,-1,-1):
        for i in range(t+1):
            call[t][i] = np.exp(-r*dt) * (p*call[t+1][i] + (1-p)*call[t+1][i+1])

    return call[0][0]

def calcOptionPriceMarkov(S0,T,M,r,K,sig):
    dt = T/M
    u,d = calcJump(sig,dt,r)
    p = (np.exp(r*dt)-d)/(u-d) 

    call = [0]*(M+1)
    for i in range(M+1):
        call[i] = max(S0*(u**i)*(d**(M-i)) - K, 0)
    for i in range(M):
        for j in range(M-i):
            call[j] = ((1-p)*call[j] + p*call[j+1])*np.exp(-r*dt)
    return call[0]

def main():
    S0 = 100
    K = 105
    T = 1
    r = 0.08
    sigma = 0.2
    M = [5,10,20,25,50,100,400]

    for m in M:
        time1 = time.time()
        p1 = calcOptionPrice(m,T,S0,r,K,sigma) 
        time2 = time.time()
        p2 = calcOptionPriceMarkov(S0,T,m,r,K,sigma)
        time3 = time.time()
        print("M = %d"%m)
        print("Price using normal method = %.6f"%p1)
        print("Price using efficient method = %.6f"%p2)
        print("Time taken by normal method = %.6f sec"%(time2-time1))
        print("Time taken by efficient method = %.6f sec"%(time3-time2))
        print("")


if __name__ == '__main__':
    main()


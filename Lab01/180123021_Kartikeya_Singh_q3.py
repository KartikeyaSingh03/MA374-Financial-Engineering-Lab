import numpy as np

# calculate u and d
def calcJump(sig,dt,r):
    u = np.exp(sig*(dt**0.5) + (r - 0.5*(sig**2))*dt)
    d = np.exp(-sig*(dt**0.5) + (r - 0.5*(sig**2))*dt)
    return u,d

def calcOptionPriceMatrix(u,d,p,M,dt,S0,r,K):
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

    return call, put		

def main():
    S0 = 100.0
    K = 105.0
    T = 5
    r = 0.05
    sig = 0.3
    M = 20

    dt = T/M   
    u,d = calcJump(sig,dt,r)
    p = (np.exp(r*dt)-d)/(u-d) 

    if d<np.exp(r*dt) and np.exp(r*dt)<u:
        call, put = calcOptionPriceMatrix(u,d,p,M,dt,S0,r,K)
    else:
        print("There is an Arbitrage Opportunity for M = %d"%M)
        return

    times = [0,0.5,1,1.5,3,4.5]

    for t in times:
        index = int(t/dt)
        print("Option values for Call options at time t = %s: \n" % t)
        for i in range(index+1):
            print("%.4f" % (call[index][i]))
        print("\n")
        print("Option values for Put options at time t = %s: \n" % t)
        for i in range(index+1):
            print("%.4f" % (put[index][i]))
        print("\n")
    
if __name__ == "__main__":
    main()
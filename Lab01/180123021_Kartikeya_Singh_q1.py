import numpy as np
import csv

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

def writeToFile(data):
    filename = "q1_data.csv"

    fields = ['M','Call Price','Put Price']

    with open(filename,'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields) 
        csvwriter.writerows(data)


def main():
    S0 = 100.0
    K = 105.0
    T = 5
    r = 0.05
    sig = 0.3
    M = [1, 5, 10, 20, 50, 100, 200, 400]

    data = []
    for m in M:
        dt = T/m   
        u,d = calcJump(sig,dt,r)
        p = (np.exp(r*dt)-d)/(u-d) 

        if d<np.exp(r*dt) and np.exp(r*dt)<u:
            callPrice, putPrice = calcOptionPrice(u,d,p,m,dt,S0,r,K)
            data.append([m,callPrice,putPrice])
        else:
            print("There is an Arbitrage Opportunity for M = %d"%m)
            return

    writeToFile(data)

if __name__ == "__main__":
    main()
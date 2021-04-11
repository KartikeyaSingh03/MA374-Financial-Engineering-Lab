import numpy as np
import time

def calcJump(sig,dt,r):
    u = np.exp(sig*(dt**0.5) + (r - 0.5*(sig**2))*dt)
    d = np.exp(-sig*(dt**0.5) + (r - 0.5*(sig**2))*dt)
    return u,d


def calcOptionPrice(S0,M,T,r,sigma):
    
    dt = T/M
    u,d = calcJump(sigma,dt,r)

    p = (np.exp(r*dt)-d)/(u-d)

    prices = []
    prices.append([S0])

    for i in range(M):
        price = []
        for s in prices[i]:
            price.append(s*u)
            price.append(s*d)
        
        prices.append(price)
        
    length = len(prices[-1])
    max_prices = [0]*length

    for i in reversed(range(M+1)):
        for j in range(length):
            max_prices[j] = max(max_prices[j],prices[i][int(j/(2**(M-i)))])

    payoff = []
    for i in range(length):
        payoff.append(max_prices[i] - prices[-1][i])

    inter_prices = []

    while(len(payoff) > 1):
        C = []
        for i in range(0,len(payoff),2):
            c = np.exp(-r*dt)*(p*payoff[i] + (1-p)*payoff[i+1]) 
            C.append(c)
        payoff = C
        inter_prices.append(payoff)

    return inter_prices

def main():
    S0 = 100
    T = 1
    r = 0.08
    sigma = 0.2
    M = [5, 10, 20]

    for m in M:
        start = time.time()
        price = calcOptionPrice(S0,m,T,r,sigma)[-1][0]
        end = time.time()
        print("M = %d\nPrice = %.6f\nTime taken = %.6f seconds\n"%(m,price,end-start))

    print("Intermediate option Prices for M = 5:")
    prices = calcOptionPrice(S0,5,T,r,sigma)
    for i in range(5):
        print("T = %d:"%i)
        prices[4-i].sort()
        for p in prices[4-i]:
            print("%.6f"%p, end = ", ") 
        print("")   

if __name__ == '__main__':
    main()

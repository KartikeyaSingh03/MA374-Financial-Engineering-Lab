import numpy as np
import time

inter_vals = {}

def calcJump(sig,dt,r):
    u = np.exp(sig*(dt**0.5) + (r - 0.5*(sig**2))*dt)
    d = np.exp(-sig*(dt**0.5) + (r - 0.5*(sig**2))*dt)
    return u,d
    
def lookback(price_dict,s,mx,n,M,u,d,p,dt,r):
    if (s,mx) in price_dict.keys():
        return price_dict[(s,mx)]
    elif n == M:
        price_dict[(s,mx)] = mx - s
        return price_dict[(s,mx)]
    else:
        up = lookback(price_dict,s*u,max(s*u,mx),n+1,M,u,d,p,dt,r)
        down = lookback(price_dict,s*d,max(s*d,mx),n+1,M,u,d,p,dt,r)
        
        price_dict[(s,mx)] = np.exp(-r*dt)*(p*up + (1-p)*down)
        return price_dict[(s,mx)]
    
def lookback2(price_dict,s,mx,n,M,u,d,p,dt,r):
    if (s,mx) in price_dict.keys():
        return price_dict[(s,mx)]
    elif n == M:
        price_dict[(s,mx)] = mx - s
        return price_dict[(s,mx)]
    else:
        up = lookback2(price_dict,s*u,max(s*u,mx),n+1,M,u,d,p,dt,r)
        down = lookback2(price_dict,s*d,max(s*d,mx),n+1,M,u,d,p,dt,r)
    
        price_dict[(s,mx)] = np.exp(-r*dt)*(p*up + (1-p)*down)
        
        if n not in inter_vals.keys():
            inter_vals[n] = []
        inter_vals[n].append(price_dict[(s,mx)])

        return price_dict[(s,mx)]

def calcOptionPrice(S0,M,T,r,sigma):
    
    dt = T/M
    u,d = calcJump(sigma,dt,r)

    p = (np.exp(r*dt)-d)/(u-d)

    price_dict = {}
    inter_vals.clear()

    return lookback(price_dict,S0,S0,0,M,u,d,p,dt,r)

def calcInterVals(S0,M,T,r,sigma):
    dt = T/M
    u,d = calcJump(sigma,dt,r)

    p = (np.exp(r*dt)-d)/(u-d)

    price_dict = {}

    return lookback2(price_dict,S0,S0,0,M,u,d,p,dt,r)

def main():
    M = [5, 10, 25, 50]
    S0 = 100
    T = 1
    r = 0.08
    sigma = 0.2

    for m in M:
        start = time.time()
        price = calcOptionPrice(S0,m,T,r,sigma)
        end = time.time()
        time_taken = end - start
        print("M = %d\nPrice = %.6f\nTime taken = %.6f seconds\n"%(m,price,time_taken))

    print("Intermediate option Prices for M = 5:")
    prices = calcInterVals(S0,5,T,r,sigma)

    for T, prices5 in inter_vals.items():
        print("T = %d:"%T)
        prices5.sort()
        for p in prices5:
            print("%.6f"%p, end = ", ") 
        print("")

if __name__ == '__main__':
    main()


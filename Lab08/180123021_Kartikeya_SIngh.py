import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import erf
plt.rcParams.update({'figure.max_open_warning': 0})

def removeNan(X):
    y = [x for x in X if not np.isnan(x)]    
    return y   

def readData(filename):
    df = pd.read_csv(filename)
    df.set_index('Date',inplace=True)
    data = df.to_dict()
    for key, vals in data.items():
        data[key] = removeNan(list(vals.values()))
    return data

def historicalVolatility(prices, duration):
    req_prices = prices[-duration:]
    
    R = []
    for i in range(1, len(req_prices)):
        ri = (req_prices[i] - req_prices[i-1])/req_prices[i-1]
        R.append(ri)

    var = np.var(R)
    sig_d = np.sqrt(var)
    sig_a = np.sqrt(252)*sig_d
    return sig_a

def d(x, tau, sig, K, r):
    d_plus = (1/(sig*np.sqrt(tau)))*(np.log(x/K) + tau*(r + (sig*sig)/2))
    d_minus = (1/(sig*np.sqrt(tau)))*(np.log(x/K) + tau*(r - (sig*sig)/2))

    return d_plus, d_minus

def N(x):
	return 0.5*(1 + erf(x/np.sqrt(2)))

def C(t, x, T, sig, K, r):
    if x == 0:
        return 0
    if t == T:
        return max(x - K, 0)

    tau = T-t
    d_plus, d_minus = d(x, tau, sig, K, r)
    
    price = x*N(d_plus) - K*np.exp(-r*tau)*N(d_minus)

    return price

def P(t, x, T, sig, K, r):
    call = C(t,x,T,sig,K,r)
    put = call + K*np.exp(-r*(T-t)) - x

    return put

bse_data = readData('bsedata1.csv')
nse_data = readData('nsedata1.csv')

def q1():
    print("\n****** Q1 ******\n")

    # Days in a month
    n_days = 20

    for company, prices in bse_data.items():
        vol = historicalVolatility(prices, n_days)
        print("Historical Volatility for %s (BSE) = %.6f"%(company, vol))

    for company, prices in nse_data.items():
        vol = historicalVolatility(prices, n_days)
        print("Historical Volatility for %s (NSE) = %.6f"%(company, vol))

def q2():
    print("\n****** Q2 ******\n")

    r = 0.05
    T = 0.5
    t = 0
    n_days = 20

    A = np.arange(0.5, 1.51, 0.1)

    for company, prices in bse_data.items():
        S0 = prices[-1]
        K = S0
        sig = historicalVolatility(prices, n_days)
        call_price = C(t, S0, T, sig, K, r)
        put_price = P(t, S0, T, sig, K, r)
        print(company, "(BSE) (K = S0)")
        print("Call Price = %.6f"%call_price)
        print("Put Price =  %.6f"%put_price)
        print("")

        data = {"K":[],"Call Price":[],"Put Price":[]}
        for a in A:
            K = a*S0
            call_price = C(t, S0, T, sig, K, r)
            put_price = P(t, S0, T, sig, K, r)
            data["K"].append(K)
            data["Call Price"].append(call_price)
            data["Put Price"].append(put_price)

        df = pd.DataFrame(data)
        df.to_csv(f"q2_{company}.csv", header=True, index=False)

    for company, prices in nse_data.items():
        S0 = prices[-1]
        K = S0
        sig = historicalVolatility(prices, n_days)
        call_price = C(t, S0, T, sig, K, r)
        put_price = P(t, S0, T, sig, K, r)
        print(company, "(NSE) (K = S0)")
        print("Call Price = %.6f"%call_price)
        print("Put Price =  %.6f"%put_price)
        print("")

        data = {"K":[],"Call Price":[],"Put Price":[]}
        for a in A:
            K = a*S0
            call_price = C(t, S0, T, sig, K, r)
            put_price = P(t, S0, T, sig, K, r)
            data["K"].append(K)
            data["Call Price"].append(call_price)
            data["Put Price"].append(put_price)

        df = pd.DataFrame(data)
        df.to_csv(f"q2_{company}.csv", header=True, index=False)   

def q3():
    
    r = 0.05
    T = 0.5
    t = 0
    n_days_month = 20

    A = np.arange(0.5, 1.51, 0.1)

    for company, prices in bse_data.items():
        months = range(1, 61)
        vols = []
        for i in range(1, 61):
            n_days = i*n_days_month
            vol = historicalVolatility(prices, n_days)
            vols.append(vol)

        plt.plot(months, vols, label = company)

    plt.xlabel("Number of Months")
    plt.ylabel("Historical Volatility")
    plt.title("Historical Volatility vs Number of Months (BSE)")
    plt.legend(loc = "best")
    plt.savefig("q3_BSE_volatility")
    plt.clf()

    for company, prices in nse_data.items():
        months = range(1, 61)
        vols = []
        for i in range(1, 61):
            n_days = i*n_days_month
            vol = historicalVolatility(prices, n_days)
            vols.append(vol)

        plt.plot(months, vols, label = company)

    plt.xlabel("Number of Months")
    plt.ylabel("Historical Volatility")
    plt.title("Historical Volatility vs Number of Months (NSE)")
    plt.legend(loc = "best")
    plt.savefig("q3_NSE_volatility")
    plt.clf()

    for company, prices in bse_data.items():
        S0 = prices[-1]
        months = range(1,61)
        fig = plt.figure(figsize=(5.6,4.2))
        for a in A:
            K = a*S0
            call_prices= []
            for i in range(1,61):
                n_days = i*n_days_month
                sig = historicalVolatility(prices, n_days)
                call = C(t, S0, T, sig, K, r)
                call_prices.append(call)

            plt.plot(months, call_prices, label = "K = %.1fS0"%a)

        plt.xlabel("Number of Months")
        plt.ylabel("Call Price")
        plt.title(f"Call Price vs Historical Volatility for {company} (BSE)")
        plt.legend(loc = "best")
        plt.savefig(f"q3_{company}_call")
        plt.clf()

    for company, prices in nse_data.items():
        S0 = prices[-1]
        months = range(1,61)
        fig = plt.figure(figsize=(5.6,4.2))
        for a in A:
            K = a*S0
            call_prices= []
            for i in range(1,61):
                n_days = i*n_days_month
                sig = historicalVolatility(prices, n_days)
                call = C(t, S0, T, sig, K, r)
                call_prices.append(call)

            plt.plot(months, call_prices, label = "K = %.1fS0"%a)

        plt.xlabel("Number of Months")
        plt.ylabel("Call Price")
        plt.title(f"Call Price vs Historical Volatility for {company} (NSE)")
        plt.legend(loc = "best")
        plt.savefig(f"q3_{company}_call")
        plt.clf()

    for company, prices in bse_data.items():
        S0 = prices[-1]
        months = range(1,61)
        fig = plt.figure(figsize=(5.6,4.2))
        for a in A:
            K = a*S0
            put_prices= []
            for i in range(1,61):
                n_days = i*n_days_month
                sig = historicalVolatility(prices, n_days)
                put = P(t, S0, T, sig, K, r)
                put_prices.append(put)

            plt.plot(months, put_prices, label = "K = %.1fS0"%a)

        plt.xlabel("Number of Months")
        plt.ylabel("Put Price")
        plt.title(f"Put Price vs Historical Volatility for {company} (BSE)")
        plt.legend(loc = "best")
        plt.savefig(f"q3_{company}_put")
        plt.clf()

    for company, prices in nse_data.items():
        S0 = prices[-1]
        months = range(1,61)
        fig = plt.figure(figsize=(5.6,4.2))
        for a in A:
            K = a*S0
            put_prices= []
            for i in range(1,61):
                n_days = i*n_days_month
                sig = historicalVolatility(prices, n_days)
                put = P(t, S0, T, sig, K, r)
                put_prices.append(put)

            plt.plot(months, put_prices, label = "K = %.1fS0"%a)

        plt.xlabel("Number of Months")
        plt.ylabel("Put Price")
        plt.title(f"Put Price vs Historical Volatility for {company} (NSE)")
        plt.legend(loc = "best")
        plt.savefig(f"q3_{company}_put")
        plt.clf()

def main():
    q1()    
    q2()
    q3()

if __name__ == '__main__':
    main()
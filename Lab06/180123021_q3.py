import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

def N(x,mu,var):
    return (1/(np.sqrt(2*var*np.pi)))*(np.exp((-0.5)*(x-mu)*(x-mu)/var))

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

bse_daily = readData('bsedata1_daily.csv')
bse_weekly = readData('bsedata1_weekly.csv')
bse_monthly = readData('bsedata1_monthly.csv')
nse_daily = readData('nsedata1_daily.csv')
nse_weekly = readData('nsedata1_weekly.csv')
nse_monthly = readData('nsedata1_monthly.csv')

def compLogReturns(S):
    R = []

    n = len(S)
    for i in range(1,n):
        ri = (S[i] - S[i-1])/S[i-1]
        R.append(ri)

    R = [np.log(1 + r) for r in R]

    mu = np.mean(R)
    sig = np.sqrt(np.var(R))

    R_cap = [((ri-mu)/sig) for ri in R]
    return R_cap

def plotGraph(company, returns, frequency, nbins, market):
    X = np.linspace(-4,4,10000)
    Y = N(X,0,1)

    fig = plt.figure(figsize=(5.6,4.2))
    plt.hist(returns, bins = nbins, edgecolor = 'black', density = 1, label = 'Log-Returns')
    plt.plot(X,Y,color = 'red', label = 'N(0, 1)')
    plt.title('%s Log-Returns for %s (%s)'%(frequency,company,market))
    plt.xlabel('Log-Returns')
    plt.ylabel('Frequency')
    plt.legend(loc = 'best')
    plt.savefig( frequency + "_" + "log_" + market + "_" + company )
    plt.clf()

def solve(data,freq,market):
    if freq == "Daily":
        nbins = 125
    elif freq == "Weekly":
        nbins = 25
    else:
        nbins = 10
    for company, prices in data.items():
        returns = compLogReturns(prices)
        plotGraph(company, returns, freq, nbins,market)

def q3():
    solve(bse_daily, "Daily", "BSE")
    solve(nse_daily, "Daily", "NSE")
    solve(bse_weekly, "Weekly", "BSE")
    solve(nse_weekly, "Weekly", "NSE")
    solve(bse_monthly, "Monthly", "BSE")
    solve(nse_monthly, "Monthly", "NSE")
    
def main():
    q3()

if __name__ == '__main__':
    main()
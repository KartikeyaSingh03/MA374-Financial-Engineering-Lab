import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

def removeNan(X):
    y = [x for x in X if not np.isnan(x)]    
    return y   

def readData(filename, freq):
    if freq == "Daily":
        end = 986
    elif freq == "Weekly":
        end = 209
    else:
        end = 48
    df = pd.read_csv(filename)
    df.set_index('Date',inplace=True)
    data = df.to_dict()
    data_till_2017 = {}
    for key, vals in data.items():
        data[key] = removeNan(list(vals.values()))
        data_till_2017[key] = removeNan(list(vals.values())[:end])
    
    return data, data_till_2017

def computeMuSigma(S):
    R = []

    n = len(S)
    for i in range(1,n):
        ri = (S[i] - S[i-1])/S[i-1]
        R.append(ri)

    R = [np.log(1 + r) for r in R]

    mu = np.mean(R)
    sig = np.sqrt(np.var(R))

    return mu,sig

def nextTerm(lamb,mu,sigma,St):
    Xt = np.log(St)
    Z = np.random.normal(0,1)
    
    deltaT = 1
    N = np.random.poisson(lamb*deltaT)

    # calculating the value of jump
    M = 0
    if N != 0:
        for i in range(N):
            Y = np.random.lognormal(mu,sigma)
            M += np.log(Y)
 
    Xt1 = Xt + (mu - (sigma**2)/2)*deltaT + sigma*Z*np.sqrt(deltaT) + M
    St1 = np.exp(Xt1)

    return St1

def predictPrice(data, freq):

    currPrice = data[-1]
    mu, sig = computeMuSigma(data)

    if freq == "Daily":
        n = 245
    elif freq == "Weekly":
        n = 52
    else:
        n = 12

    prices = []
    for i in range(n):
        nextPrice = nextTerm(0.1,mu,sig,currPrice)
        prices.append(nextPrice)
        currPrice = nextPrice
    
    return prices

def solve(data_actual, data, freq, market):
    for company, prices in data.items():
        predicted_prices = predictPrice(prices, freq)
        prices.extend(predicted_prices)
        
        X = range(len(prices))

        fig = plt.figure(figsize=(5.6,4.2))
        plt.plot(X, prices, label = 'Predicted')
        plt.plot(X, data_actual[company], label = 'Actual')
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title(f"{freq} Price for {company} ({market})")
        plt.legend(loc = 'best')
        plt.savefig(f"Predicted_{freq}_{market}_{company}")
        plt.clf()

def q4():
    bse_daily_actual, bse_daily_train = readData('bsedata1_daily.csv',"Daily")
    bse_weekly_actual, bse_weekly_train = readData('bsedata1_weekly.csv', "Weekly")
    bse_monthly_actual, bse_monthly_train = readData('bsedata1_monthly.csv', "Monthly")
    nse_daily_actual, nse_daily_train = readData('nsedata1_daily.csv',"Daily")
    nse_weekly_actual, nse_weekly_train = readData('nsedata1_weekly.csv', "Weekly")
    nse_monthly_actual, nse_monthly_train = readData('nsedata1_monthly.csv', "Monthly")

    solve(bse_daily_actual, bse_daily_train, "Daily", "BSE")
    solve(nse_daily_actual,nse_daily_train, "Daily", "NSE")
    solve(bse_weekly_actual,bse_weekly_train, "Weekly", "BSE")
    solve(nse_weekly_actual,nse_weekly_train, "Weekly", "NSE")
    solve(bse_monthly_actual,bse_monthly_train, "Monthly", "BSE")
    solve(nse_monthly_actual,nse_monthly_train, "Monthly", "NSE")

def main():
    q4()

if __name__ == '__main__':
    main()
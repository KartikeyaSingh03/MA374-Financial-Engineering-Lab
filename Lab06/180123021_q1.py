import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def readData(filename):
    df = pd.read_csv(filename)
    df.set_index('Date',inplace=True)
    data = df.to_dict()
    for key, vals in data.items():
        data[key] = list(vals.values())
    return data

bse_daily = readData('bsedata1_daily.csv')
bse_weekly = readData('bsedata1_weekly.csv')
bse_monthly = readData('bsedata1_monthly.csv')
nse_daily = readData('nsedata1_daily.csv')
nse_weekly = readData('nsedata1_weekly.csv')
nse_monthly = readData('nsedata1_monthly.csv')

def plotPriceVsTime(data,title,freq):
    for company, prices in data.items():
        X = range(len(prices))
        Y = prices
        plt.plot(X,Y,label = company)

    plt.xlabel(freq)
    plt.ylabel('Price')
    plt.title(title)
    plt.legend(loc = 'upper left')
    plt.savefig(title)
    plt.clf()

def q1():
    plotPriceVsTime(bse_daily, "BSE Daily", "Day")
    plotPriceVsTime(bse_weekly, "BSE Weekly", "Week")
    plotPriceVsTime(bse_monthly, "BSE Monthly", "Month")
    plotPriceVsTime(nse_daily, "NSE Daily", "Day")
    plotPriceVsTime(nse_weekly, "NSE Weekly", "Week")
    plotPriceVsTime(nse_monthly, "NSE Monthly", "Month")

def main():
    q1()

if __name__ == '__main__':
    main()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import erf, sqrt, log, exp
from mpl_toolkits.mplot3d import Axes3D


from pandas.io.pytables import DataCol

files = ['NIFTYoptiondata.csv', 'stockoptiondata_CIPLA.csv',  'stockoptiondata_COALINDIA.csv',  'stockoptiondata_ICICI.csv' , 'stockoptiondata_ITC.csv']
companies = ['NIFTY', 'CIPLA', 'COALINDIA', 'ICICI', 'ITC']

def readData(filename):
    df = pd.read_csv(filename)

    call_data = []
    put_data = []

    for index, row in df.iterrows():
        call_price = row['Call Price']
        put_price = row['Put Price']
        strike_price = row['Strike Price']
        maturity = (60 - (index%61))%61
        call_data.append({'Price': call_price, 'Strike': strike_price, 'Maturity':maturity})
        put_data.append({'Price': put_price, 'Strike': strike_price, 'Maturity':maturity})

    return call_data, put_data

def d(x, tau, sig, K, r):
    d_plus = (1/(sig*sqrt(tau)))*(log(x/K) + tau*(r + (sig*sig)/2))
    d_minus = (1/(sig*sqrt(tau)))*(log(x/K) + tau*(r - (sig*sig)/2))

    return d_plus, d_minus

def N(x):
	return 0.5*(1 + erf(x/sqrt(2)))

def C(t, x, T, sig, K, r):
    if x == 0:
        return 0
    if t == T:
        return max(x - K, 0)

    tau = T-t
    d_plus, d_minus = d(x, tau, sig, K, r)
    
    price = x*N(d_plus) - K*exp(-r*tau)*N(d_minus)

    return price

def P(t, x, T, sig, K, r):
    call = C(t,x,T,sig,K,r)
    put = call + K*exp(-r*(T-t)) - x

    return put

def impliedVolatility(call_option_price,put_option_price, s, K, r, T):
    eps = 1e-6
    t = 0 

    # Call Option 
    upper_vol = 100.0
    max_vol = 100.0
    lower_vol = 0.0000001
    call_price = np.inf

    while abs(call_price - call_option_price) >= eps:
        mid_vol = (upper_vol + lower_vol)/2

        call_price = C(0, s, T, mid_vol, K, r)
        lower_price = C(0, s, T, lower_vol, K, r)

        if (lower_price - call_option_price) * (call_price - call_option_price) > 0:
            lower_vol = mid_vol
        else:
            upper_vol = mid_vol

        if mid_vol > max_vol - 0.01:
            mid_vol = 1e-6
            break

    call_vol = mid_vol

    # Put Price
    upper_vol = 100.0
    lower_vol = 0.000001
    min_vol = 1e-5
    put_price = np.inf

    while abs(put_price - put_option_price) >= eps:
        mid_vol = (upper_vol + lower_vol)/2
        put_price = P(0, s, T, mid_vol, K, r)
        
        upper_price = P(0, s, T, upper_vol, K, r)

        if (upper_price - put_option_price) * (put_price - put_option_price) > 0:
            upper_vol = mid_vol
        else:
            lower_vol = mid_vol

        if mid_vol > max_vol - 0.01 or mid_vol < min_vol:
            mid_vol = 1e-6
            break

    put_vol = mid_vol

    return call_vol, put_vol

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

def plot2D(data, optType, fixedMaturity, fixedStrike, company):
    
    # Maturity vs Option Price

    for strike in fixedStrike:
        maturity = []
        prices = []
        for row in data:
            if row['Strike'] == strike:
                mat = row['Maturity']
                price = row['Price']
                maturity.append(mat)
                prices.append(price)

        plt.plot(maturity, prices, label = 'Strike Price = %d'%strike)
    
    plt.xlabel("Maturity")
    plt.ylabel("%s Option Price")
    plt.title("%s Option Price vs Maturity for %s"%(optType, company))
    plt.legend(loc = 'best')
    plt.savefig(f"q2_{company}_{optType}_maturity")
    plt.clf()

    # Strike Price vs Option Price 

    for fixedMat in fixedMaturity:
        strikePrices = []
        prices = []
        for row in data:
            if row['Maturity'] == fixedMat:
                strike = row['Strike']
                price = row['Price']
                strikePrices.append(strike)
                prices.append(price)

        plt.plot(strikePrices, prices, label = 'Maturity = %d'%fixedMat)

    plt.xlabel("Strike Price")
    plt.ylabel("%s Option Price")
    plt.title("%s Option Price vs Strike Price for %s"%(optType, company))
    plt.legend(loc = 'best')
    plt.savefig(f"q2_{company}_{optType}_strike")
    plt.clf()

def plot3D(data,optType, company):
    maturity = []
    prices = []
    strikePrices = []

    for row in data:
        mat = row['Maturity']
        price = row['Price']
        strike = row['Strike']
        maturity.append(mat)
        prices.append(price)
        strikePrices.append(strike)

    ax = plt.axes(projection='3d')
    ax.scatter(np.array(maturity), np.array(strikePrices), np.array(prices))
    ax.set_xlabel("Maturity")
    ax.set_ylabel("Strike Price")
    ax.set_zlabel("%s Option Price"%optType) 
    ax.set_title("%s Option Price for %s"%(optType, company))
    plt.savefig(f"q2_{company}_{optType}_3D")
    plt.clf()
 

def q2():
    fixedMaturity = [1, 30, 60]

    for filename,company in zip(files,companies):
        call_data, put_data  = readData(filename)

        n = len(call_data)
        fixedStrike = [call_data[0]['Strike'], call_data[-1]['Strike'], call_data[n//2]['Strike']]

        plot3D(call_data, "Call", company)
        plot3D(put_data, "Put", company)
        plot2D(call_data, "Call", fixedMaturity, fixedStrike, company)
        plot2D(put_data, "Put", fixedMaturity, fixedStrike, company)

def q3():

    df = pd.read_csv('nsedata1.csv')

    r = 0.05

    for filename, company in zip(files, companies):
        call_data, put_data = readData(filename)
        fixed_strike = call_data[0]['Strike']
        fixed_maturity = 30

        call_vols = []
        put_vols = []
        mats = []
        strikes = []
        call_vols2D = []
        put_vols2D = []
        mats2D = []
        call_vols2D1 = []
        put_vols2D1 = []
        strikes2D1 = []
        

        for row_call, row_put in zip(call_data, put_data):
            mat = row_call['Maturity']
            if 0 < mat < 60:
                T = mat/252
                K = row_call['Strike']
                call_option_price = row_call['Price']
                put_option_price = row_put['Price']
                s = float(df[company][59-mat])
                call_vol, put_vol = impliedVolatility(call_option_price, put_option_price, s, K, r, T)
                call_vols.append(call_vol)
                put_vols.append(put_vol)
                mats.append(mat)
                strikes.append(K)
                if K == fixed_strike:
                    call_vols2D.append(call_vol)
                    put_vols2D.append(put_vol)
                    mats2D.append(mat)
                if mat == fixed_maturity:
                    call_vols2D1.append(call_vol)
                    put_vols2D1.append(put_vol)
                    strikes2D1.append(K)

        plt.scatter(mats2D, call_vols2D, label = 'Call', s = 1)
        plt.scatter(mats2D, put_vols2D, label = 'Put', s = 1)
        plt.legend(loc = 'best')
        plt.title(company)
        plt.xlabel('Maturity')
        plt.ylabel('Implied Volatility')
        plt.title(f"Implied Volatility vs Maturity for {company} (Strike Price = {fixed_strike})")
        plt.savefig(f"q3_{company}_volatility_2D_maturity")
        plt.clf()

        plt.plot(strikes2D1, call_vols2D1, label = 'Call')
        plt.plot(strikes2D1, put_vols2D1, label = 'Put')
        plt.legend(loc = 'best')
        plt.title(company)
        plt.xlabel('Strike Price')
        plt.ylabel('Implied Volatility')
        plt.title(f"Implied Volatility vs Strike Price for {company} (Maturity = {fixed_maturity})")
        plt.savefig(f"q3_{company}_volatility_2D_strike")
        plt.clf()

        ax = plt.axes(projection='3d')
        ax.scatter(np.array(mats), np.array(strikes), np.array(call_vols))
        ax.set_xlabel("Maturity")
        ax.set_ylabel("Strike Price")
        ax.set_zlabel("Implied Volatility") 
        ax.set_title(f"Call Implied Volatility for {company}")
        plt.savefig(f"q3_{company}_volatility_call_3D")
        plt.clf()

        ax = plt.axes(projection='3d')
        ax.scatter(np.array(mats), np.array(strikes), np.array(put_vols))
        ax.set_xlabel("Maturity")
        ax.set_ylabel("Strike Price")
        ax.set_zlabel("Implied Volatility") 
        ax.set_title(f"Put Implied Volatility for {company}")
        plt.savefig(f"q3_{company}_volatility_put_3D")
        plt.clf()

def q4():

    df = pd.read_csv('nsedata1.csv')

    for company in companies:

        prices = list(df[company])
        vols = []
        T = range(2, 59)

        for t in T:
            vol = historicalVolatility(prices, t)
            vols.append(vol)

        plt.plot(T, vols)
        plt.xlabel("Maturity")
        plt.ylabel("Historical Volatility")
        plt.title("Historical Volatility vs Maturity for %s"%company)
        plt.savefig(f"q4_{company}_historical_volatility")
        plt.clf()

def main():
    q2()
    q3()
    q4()

if __name__ == '__main__':
    main()

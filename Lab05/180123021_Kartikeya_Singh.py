import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def readData(filename):
    df = pd.read_csv(filename)
    df.set_index('Date',inplace=True)
    df = df.pct_change()
    M = np.array(np.mean(df,axis=0)[:])
    M = (1+M)**12 - 1
    C = np.array(df.cov())*12    
    return M,C 

def readData1(filename):
    df = pd.read_csv(filename,usecols=['Adj Close'])
    df = df.pct_change()
    data = np.array(df['Adj Close'][1:])
    mu_market = (np.mean(data)+1)**12 - 1
    sigma_market = np.sqrt(12 * np.var(data))
    return mu_market,sigma_market

def minVarPortfolio(m,C):
    Cinv = np.linalg.inv(C)
    u = np.array([1]*len(m))

    uT = np.transpose(u)
    mT = np.transpose(m)

    # @ operator represents matrix multiplication.
    wmvp = (u @ Cinv)/ (u @ Cinv @ uT)
    wmvpT = np.transpose(wmvp)
    
    mumvp = (wmvp @ mT)
    varmvp = (wmvp @ C @ wmvpT)
    sigmamvp = np.sqrt(varmvp)

    return wmvp, mumvp, sigmamvp

def minVariance(mu,M,C):
    Cinv = np.linalg.inv(C)
    u = np.array([1]*len(M))

    a = M @ Cinv @ M
    b = u @ Cinv @ M
    c = u @ Cinv @ u

    det1 = np.linalg.det([[1, b],[mu, a]])
    det2 = np.linalg.det([[c,1],[b,mu]])
    det3 = np.linalg.det([[c,b],[b,a]])

    w = (1/det3)*(det1*(u @ Cinv) + det2*(M @ Cinv))
    wT = np.transpose(w)

    sig = np.sqrt(w @ C @ wT)

    return sig, w

def marketPortfolio(mu_rf,M,C):
    Cinv = np.linalg.inv(C)
    u = np.array([1]*len(M))
    uT = np.transpose(u)
    mT = np.transpose(M)
    Md = (M - mu_rf*u)
    
    w = (Md @ Cinv)/(Md @ Cinv @ uT)
    wT = np.transpose(w)

    mu = w @ mT
    var = w @ C @ wT
    sigma = np.sqrt(var)

    return w, mu, sigma

def q1(M, C, market,mu_ind = 0, sig_ind = 0):
    print("\n", market, "\n")

    w_mvp,mu_mvp,sig_mvp = minVarPortfolio(M,C)

    print("Minimum Variance Portfolio: ")
    print("Return (mu) = %.6f"%mu_mvp)
    print("Risk (sigma) = %.6f"%sig_mvp)

    mu_frontier = np.linspace(-0.2, 0.5, 1000)
    sigma_frontier = []

    for mu in mu_frontier:
        sig, w = minVariance(mu,M,C)
        sigma_frontier.append(sig)

    mu_rf = 0.05

    w_market,mu_market,sigma_market = marketPortfolio(mu_rf,M,C)

    sigma_CML = np.linspace(0.05,0.45,5000)
    mu_CML = []

    for sigma in sigma_CML:
        mu = mu_rf + (mu_market - mu_rf)*sigma/sigma_market
        mu_CML.append(mu)

    slope = (mu_market - mu_rf)/sigma_market
    print("Equation of Capital Market Line:")
    print("mu = %.2f + %.4f*sigma"%(mu_rf,slope))

    if market != "Not in Index":
        mu_CML1 = []

        for sigma in sigma_CML:
            mu = mu_rf + (mu_ind - mu_rf)*sigma/sig_ind
            mu_CML1.append(mu)   

        slope = (mu_ind - mu_rf)/sig_ind
        print("Equation of Capital Market Line:")
        print("mu = %.2f + %.4f*sigma"%(mu_rf,slope))

        plt.plot(sigma_CML,mu_CML1,color = 'purple',label = 'Capital Market Line with Index as MP')
        plt.scatter(sig_ind,mu_ind,color = 'brown',s = 15,label = 'Index')
     
    plt.plot(sigma_frontier,mu_frontier,color = 'blue',label = 'Markowitz Efficient Frontier')
    plt.plot(sigma_CML, mu_CML,color = 'green',label = 'Capital Market Line')
    plt.scatter(sig_mvp,mu_mvp,color = 'red',s = 10,label = 'Minimum Variance Portfolio')
    plt.scatter(sigma_market, mu_market, s = 10, color = 'brown',label = 'Market Portfolio')
    plt.title('Markowitz Efficient Frontier and CML for %s'%market)
    plt.xlabel('sigma (risk)')
    plt.ylabel('mu (return)')
    plt.legend(loc = 'best')
    plt.show()
    plt.clf()

def q2(M, C, market, mu_ind = 0):
    print("\n", market, "\n")

    mu_rf = 0.05

    w_mvp,mu_mvp,sig_mvp = minVarPortfolio(M,C)
    w_market,mu_market,sigma_market = marketPortfolio(mu_rf,M,C)

    beta_v = np.linspace(-2,2,5000)
    mu_v = []
    
    
    for beta in beta_v:
        mu1 = mu_rf + beta*(mu_market - mu_rf)
        mu_v.append(mu1)

    print("Equation of Security Market Line with calculated Market Portfolio:")
    print("mu = %.2f + %.4f*sigma"%(mu_rf,mu_market - mu_rf))

    if market != "Not in Index":
        mu_v1 = []

        for beta in beta_v:
            mu1 = mu_rf + beta*(mu_ind - mu_rf)
            mu_v1.append(mu1)

        print("Equation of Security Market Line assuming index as Market Portfolio:")
        print("mu = %.2f + %.4f*sigma"%(mu_rf,mu_market - mu_rf))

        plt.plot(beta_v,mu_v1,color = 'brown',label = 'Security Market Line with Index as MP')
        plt.scatter(1,mu_ind,color = 'purple',s = 15,label = 'Index')

    plt.plot(beta_v,mu_v,color = 'red',label = 'Security Market Line with Calulated MP')
    plt.scatter(1,mu_market,color = 'blue',s = 15,label = 'Calculated Market Portfolio')
    plt.scatter(0,mu_rf,s = 15,color = 'green',label = 'Risk Free Portfolio')    
    plt.title('Security Market Line for %s'%market)
    plt.xlabel('Beta (Systematic Risk)')
    plt.ylabel('mu (Return)')
    plt.legend(loc = 'best')
    plt.grid()
    plt.show()
    plt.clf()

def q3_bse(sigma_market_bse):
    print("\nBSE \n")

    df = pd.read_csv('BSE.csv',usecols=['Adj Close'])
    df = df.pct_change()
    sensex = np.array(df['Adj Close'][1:])
    
    df_bse = pd.read_csv('bsedata1.csv')
    df_bse.set_index('Date',inplace=True)
    cols = df_bse.columns
    df_bse = df_bse.pct_change()
    
    df_bse = df_bse.drop(df_bse.index[0])
    df_bse.insert(loc=0,column='sensex',value=sensex)
    
    cov = np.array(df_bse.cov())*12
    cov = cov/sigma_market_bse**2
    beta_bse = cov[0][1:]

    print("Values of Beta (BSE):\n")
    for comp, beta in zip(cols, beta_bse):
        print("%s: %.6f"%(comp,beta))

def q3_nse(sigma_market_nse):
    print("\nNSE \n")

    df = pd.read_csv('NSE.csv',usecols=['Adj Close'])
    df = df.pct_change()
    nifty = np.array(df['Adj Close'][1:])
    
    df_nse = pd.read_csv('nsedata1.csv')
    df_nse.set_index('Date',inplace=True)
    cols = df_nse.columns
    df_nse = df_nse.pct_change()

    df_nse = df_nse.drop(df_nse.index[0])
    df_nse.insert(loc=0,column='nifty',value=nifty)
    
    cov = np.array(df_nse.cov())*12
    cov = cov/sigma_market_nse**2
    beta_nse = cov[0][1:]

    print("Values of Beta (NSE):\n")
    for comp, beta in zip(cols, beta_nse):
        print("%s: %.6f"%(comp,beta))
    
def main():

    M_bse, C_bse = readData('bsedata1.csv')
    M_nse, C_nse = readData('nsedata1.csv')
    M_noind,C_noind = readData('noindex.csv')

    mu_market_bse, sig_market_bse = readData1('BSE.csv')
    mu_market_nse, sig_market_nse = readData1('NSE.csv')

    print(" ****** Question 1 ****** \n")

    q1(M_bse, C_bse, "BSE",mu_market_bse, sig_market_bse)
    q1(M_nse, C_nse, "NSE", mu_market_nse, sig_market_nse)
    q1(M_noind, C_noind, "Not in Index")

    print("\n ****** Question 2 ****** \n")

    q2(M_bse, C_bse, "BSE",mu_market_bse)
    q2(M_nse, C_nse, "NSE",mu_market_nse)
    q2(M_noind, C_noind, "Not in Index")

    print("\n ****** Question 3 ****** \n")

    q3_bse(sig_market_bse)
    q3_nse(sig_market_nse)    

if __name__ == '__main__':
    main()
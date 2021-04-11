import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def readData():
    df = pd.read_csv('data.csv')
    df.set_index('Date',inplace=True)
    df = df.pct_change()
    M = np.mean(df,axis=0)*12
    C = df.cov()
    cols = df.head()
    return cols,M,C   

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

def qA(M,C):
    print(" ****** Question 3-A ****** \n")

    w_mvp,mu_mvp,sig_mvp = minVarPortfolio(M,C)

    print("Minimum Variance Portfolio: ")
    print("Return (mu) = %.6f"%mu_mvp)
    print("Risk (sigma) = %.6f"%sig_mvp)

    mu_frontier = np.linspace(-0.2, 0.5, 1000)
    sigma_frontier = []

    for mu in mu_frontier:
        sig, w = minVariance(mu,M,C)
        sigma_frontier.append(sig)

    plt.plot(sigma_frontier,mu_frontier,color = 'blue',label = 'Markowitz Efficient Frontier')
    plt.scatter(sig_mvp,mu_mvp,color = 'red',s = 10,label = 'Minimum Variance Portfolio')
    plt.title('Markowitz Efficient Frontier')
    plt.xlabel('sigma (risk)')
    plt.ylabel('mu (return)')
    plt.legend(loc = 'upper left')
    plt.show()
    plt.clf()

def qB(M,C,cols):

    print("\n ****** Question 3-B ****** \n")

    mu_rf = 0.01
    _,mu_mvp,_ = minVarPortfolio(M,C)

    if mu_rf > mu_mvp:
        print("Market Portfolio Doesn't Exists.")
        return 

    w,mu,sigma = marketPortfolio(mu_rf,M,C)
    print("The Market Portfolio is: ")
    print("Return (mu) = %s"%mu)
    print("Risk (sigma) = %s"%sigma)
    print("Weights:")
    for stock,weight in zip(cols,w):
        print("(%s,%s)"%(stock,weight))

    

def qC(M,C):
    print("\n ****** Question 3-C ****** \n")

    mu_rf = 0.01
    _,mu_mvp,_ = minVarPortfolio(M,C)

    if mu_rf > mu_mvp:
        print("Market Portfolio Doesn't Exists.")
        return

    w_market,mu_market,sigma_market = marketPortfolio(mu_rf,M,C)

    sigma_CML = np.linspace(0,0.1,5000)
    mu_CML = []

    for sigma in sigma_CML:
        mu = mu_rf + (mu_market - mu_rf)*sigma/sigma_market
        mu_CML.append(mu)

    slope = (mu_market - mu_rf)/sigma_market
    print("Equation of Capital Market Line:")
    print("mu = %.2f + %.4f*sigma"%(mu_rf,slope))

    plt.plot(sigma_CML, mu_CML,color = 'blue',label = 'Capital Market Line')
    plt.scatter(sigma_market, mu_market, s = 10, color = 'red',label = 'Market Portfolio')
    plt.title('Capital Market Line')
    plt.xlabel('sigma (risk)')
    plt.ylabel('mu (return)')
    plt.legend(loc = 'upper left')
    plt.show()
    plt.clf()

    mu_frontier = np.linspace(-0.2,0.5,1000)
    sigma_frontier = []

    for mu in mu_frontier:
        sig, w = minVariance(mu,M,C)
        sigma_frontier.append(sig)

    plt.plot(sigma_CML, mu_CML,color = 'blue',label = 'Capital Market Line')
    plt.plot(sigma_frontier,mu_frontier,color = 'green',label = 'Markowitz Bullet')
    plt.scatter(sigma_market, mu_market, s = 10, color = 'red',label = 'Market Portfolio')
    plt.title('Capital Market Line with Markowitz Bullet')
    plt.xlabel('sigma (risk)')
    plt.ylabel('mu (return)')
    plt.legend(loc = 'upper left')
    plt.show()
    plt.clf()

def qD(M,C):
    print("\n ****** Question 3-D ****** \n")

    mu_rf = 0.01
    _,mu_mvp,_ = minVarPortfolio(M,C)

    if mu_rf > mu_mvp:
        print("Market Portfolio Doesn't Exists.")
        return

    w_market,mu_market,sigma_market = marketPortfolio(mu_rf,M,C)

    beta_v = np.linspace(-2,2,5000)
    mu_v = []

    for beta in beta_v:
        mu = mu_rf + beta*(mu_market - mu_rf)
        mu_v.append(mu)

    print("Equation of Security Market Line:")
    print("mu = %.2f + %.4f*sigma"%(mu_rf,mu_market - mu_rf))

    plt.plot(beta_v,mu_v,color = 'red',label = 'Security Market Line')
    plt.scatter(1,mu_market,color = 'blue',s = 15,label = 'Market Portfolio')
    plt.scatter(0,mu_rf,s = 15,color = 'green',label = 'Risk Free Portfolio')    
    plt.title('Security Market Line')
    plt.xlabel('Beta (Systematic Risk)')
    plt.ylabel('mu (Return)')
    plt.legend(loc = 'best')
    plt.show()
    plt.clf()

def main():

    cols,M,C = readData()

    print("The stocks taken are - ")
    print(list(cols))

    print("\nThe mean is given by - ")
    print(M)

    print("\nThe covariance is given by - ")
    print(C)

    M = np.array(M)
    C = np.array(C)
    cols = list(cols)

    qA(M,C)
    qB(M,C,cols)
    qC(M,C)
    qD(M,C)
    

if __name__ == '__main__':
    main()

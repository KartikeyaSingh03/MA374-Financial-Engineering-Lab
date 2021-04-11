import numpy as np
import matplotlib.pyplot as plt

def minVarPortfolio(m,C):
    Cinv = np.linalg.inv(C)
    u = np.array([1, 1, 1])

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
    u = np.array([1, 1, 1])

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
    u = np.array([1, 1, 1])
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
    print(" ****** Question 1-A ****** \n")

    w, muMVP, sigmaMVP = minVarPortfolio(M,C)

    print("Minimum Variance Portfolio:")
    print("Weights :", w)
    print("Return (mu) : ",muMVP)
    print("Risk (sigma): ",sigmaMVP)

    mu_frontier = np.linspace(0,0.5,1000)
    sigma_frontier = []

    for mu in mu_frontier:
        sig, w = minVariance(mu,M,C)
        sigma_frontier.append(sig)

    plt.plot(sigma_frontier,mu_frontier,color = 'blue',label = 'Markowitz Efficient Frontier')
    plt.scatter(sigmaMVP,muMVP,s = 10, color = 'red', label = 'Minimum Variance Portfolio')
    plt.title('Markowitz Efficient Frontier')
    plt.xlabel('sigma (risk)')
    plt.ylabel('mu (return)')
    plt.legend(loc = 'upper left')
    plt.show()
    plt.clf()

def qB(M,C):
    print("\n ****** Question 1-B ****** \n")

    mu_frontier = np.linspace(0,0.45,10)

    for mu in mu_frontier:
        sig, w = minVariance(mu,M,C)
        print("mu = %.2f, sigma = %.6f, weights = %s"%(mu,sig,w))

def qC(M,C):
    print("\n ****** Question 1-C ****** \n")

    mu_frontier = np.linspace(0,0.5,1000)

    _, muMVP, _ = minVarPortfolio(M,C)

    minRet = 0
    maxRet = 0 
    w_minRet = []
    w_maxRet = []
    epsMin = 100.0
    epsMax = 100.0
    sigReq = 0.15

    for mu in mu_frontier:
        sig, w = minVariance(mu,M,C)
        dis = abs(sig - sigReq)
        if mu <= muMVP:
            if dis < epsMin:
                epsMin = dis
                minRet = mu
                w_minRet = w
        else:
            if dis < epsMax:
                epsMax = dis
                maxRet = mu
                w_maxRet = w

    print("Assuming 15% risk: ")
    print("Minimum Return = %.6f for the portfolio = %s"%(minRet,w_minRet))
    print("Maximum Return = %.6f for the portfolio = %s"%(maxRet,w_maxRet))

def qD(M,C):
    print("\n ****** Question 1-D ****** \n")

    mu = 0.18
    sig, w = minVariance(mu,M,C)
    print("Minimum Risk Portfolio:")
    print("mu = %.2f, sigma = %.6f, weights = %s"%(mu,sig,w))

def qE(M,C):
    print("\n ****** Question 1-E ****** \n")

    mu_rf = 0.10
    
    w_market, mu_market, sigma_market = marketPortfolio(mu_rf,M,C)
    
    print("Market Portfolio: ")
    print("Return (mu) = %.6f"%mu_market)
    print("Risk (sigma) = %.6f"%sigma_market)
    print("Weights = %s"%w_market)

    sigma_CML = np.linspace(0,0.5,1000)
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

    mu_frontier = np.linspace(0,0.5,1000)
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

def qF(M,C):
    print("\n ****** Question 1-F ****** \n")

    mu_rf = 0.10
    
    w_market, mu_market, sigma_market = marketPortfolio(mu_rf,M,C)

    Sigma = [0.10, 0.25]

    for sigma in Sigma:
        mu = mu_rf + ((mu_market - mu_rf)/sigma_market)*sigma
        w_rf = (mu - mu_market)/(mu_rf - mu_market)
        w = (1 - w_rf) * w_market    

        print("Portfolio for Risk = %d %% is: "%(sigma*100))
        print("Return = %s"%mu)
        print("Risk = %s"%sigma)
        print("Weight of Risk Free Asset = %s"%w_rf)
        print("Weights of Risky Assets = %s\n"%w)

def main():
    M = [0.1, 0.2, 0.15]
    C = [
         [0.005, -0.010, 0.004],
         [-0.010, 0.040, -0.002],
         [0.004, -0.002, 0.023]   
        ]

    M = np.array(M)
    C = np.array(C)

    qA(M,C)
    qB(M,C)
    qC(M,C)
    qD(M,C)
    qE(M,C)
    qF(M,C)

if __name__ == '__main__':
    main()

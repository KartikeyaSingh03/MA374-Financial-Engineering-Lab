import numpy as np
import matplotlib.pyplot as plt

def nextStockPrice(mu,sigma,S0):
    delta_t = 1/252
    W = np.random.normal(0, 1)
    drift = delta_t * (mu - (sigma**2)/2)
    diffusion = sigma*W*np.sqrt(delta_t)
    Sk = S0 * np.exp(drift + diffusion)
    return Sk

def estimateStockPrices(mu, sigma, S0, T):
    prices = [S0]
    curr = S0
    for i in range(T):
        curr = nextStockPrice(mu, sigma, curr)
        prices.append(curr)

    return prices

def payoff(mu,sigma,K,S0,N):
    Sk = S0
    Sum = 0
    for i in range(1,N+1):
        Sk = nextStockPrice(mu,sigma,Sk)
        Sum += Sk
    meanStockPrice = Sum/(N+1)	
    Put_payoff = max(K - meanStockPrice, 0)
    Call_payoff = max(meanStockPrice - K, 0)
    return Call_payoff, Put_payoff

def calcAsianOptionPrice(mu,sigma,K,S0,r,T):
    CallPayoff = []
    PutPayoff = []
    for i in range(100):
        Call_payoff, Put_payoff = payoff(mu,sigma,K,S0,T)
        CallPayoff.append(Call_payoff)
        PutPayoff.append(Put_payoff)
    
    CallPayoff = np.array(CallPayoff)
    PutPayoff = np.array(PutPayoff)
    CallOptionPrice = CallPayoff * np.exp(-r*T/252)
    PutOptionPrice = PutPayoff * np.exp(-r*T/252)

    return np.mean(CallOptionPrice), np.mean(PutOptionPrice)

def simulatePaths():
    S0 = 100
    mu = 0.1
    sigma = 0.2
    r = 0.05
    T = 126
    dates = range(T+1)

    for i in range(10):
        prices = estimateStockPrices(mu, sigma, S0, T)
        plt.plot(dates, prices)

    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Stock prices using GBM in real world')
    plt.savefig("q1_Stock_real")
    plt.clf()

    for i in range(10):
        prices = estimateStockPrices(r, sigma, S0, T)
        plt.plot(dates, prices)

    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Stock prices using GBM in risk-neutral world')
    plt.savefig("q1_Stock_risk_neutral")
    plt.clf()


def asianOption():
    S0 = 100
    r = 0.05
    sigma = 0.2
    T = 126
    K = [90, 105, 110]

    for k in K:
        call, put = calcAsianOptionPrice(r, sigma, k, S0, r, T)

        print("Call Price for K = %d is %.6f"%(k,call))
        print("Put Price for K = %d is %.6f"%(k,put))

def sensitivityAnalysis():
    S0 = 100
    r = 0.05
    sigma = 0.2
    T = 126
    K = 105

    # Varying K
    K1 = np.linspace(85, 115, 50)
    call_prices = []
    put_prices = []

    for k1 in K1:   
        call, put = calcAsianOptionPrice(r, sigma, k1, S0, r, T)
        call_prices.append(call)
        put_prices.append(put)

    plt.plot(K1, call_prices, label = "Call")
    plt.plot(K1, put_prices, label = "Put")
    plt.xlabel("K")
    plt.ylabel("Option Price")
    plt.title("Varying K")
    plt.legend(loc = 'best')
    plt.savefig("q1_K")
    plt.clf()

    # Varying sigma
    Sigma1 = np.linspace(0.05, 0.35, 50)
    call_prices = []
    put_prices = []

    for sigma1 in Sigma1:   
        call, put = calcAsianOptionPrice(r, sigma1, K, S0, r, T)
        call_prices.append(call)
        put_prices.append(put)

    plt.plot(Sigma1, call_prices, label = "Call")
    plt.plot(Sigma1, put_prices, label = "Put")
    plt.xlabel("sigma")
    plt.ylabel("Option Price")
    plt.title("Varying Sigma")
    plt.legend(loc = 'best')
    plt.savefig("q1_sigma")
    plt.clf()

    # Varying S0
    S1 = np.linspace(85, 115, 50)
    call_prices = []
    put_prices = []

    for s1 in S1:   
        call, put = calcAsianOptionPrice(r, sigma, K, s1, r, T)
        call_prices.append(call)
        put_prices.append(put)

    plt.plot(S1, call_prices, label = "Call")
    plt.plot(S1, put_prices, label = "Put")
    plt.xlabel("S0")
    plt.ylabel("Option Price")
    plt.title("Varying S0")
    plt.legend(loc = 'best')
    plt.savefig("q1_s0")
    plt.clf()    

def main():
    simulatePaths()
    asianOption()
    sensitivityAnalysis()

if __name__ == '__main__':
    main()
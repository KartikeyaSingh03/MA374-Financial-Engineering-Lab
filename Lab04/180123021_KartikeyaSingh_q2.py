import numpy as np
import matplotlib.pyplot as plt

def minVariance(mu,M,C,n):
    Cinv = np.linalg.inv(C)
    u = np.array([1]*n)

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

def noShort(W):
    for w in W:
        if w < 0 or w > 1:
            return False
    return True

# removes index ind from M, C
def getMC(M,C,ind):
    n = len(M)
    M1 = []
    C1 = []
    
    for i in range(n):
        if i != ind:
            M1.append(M[i])

    for i in range(n):
        if i != ind:
            row = []
            for j in range(n):
                if j != ind:
                    row.append(C[i][j])
            C1.append(row)
    
    return np.array(M1), np.array(C1)

def calcFrontier(M,C,n):

    mu_frontier = np.linspace(0,0.25,1000)
    sigma_frontier = []

    for mu in mu_frontier:
        sigma, w = minVariance(mu,M,C,n)
        sigma_frontier.append(sigma)

    return mu_frontier, sigma_frontier

def calcFrontierNoShort(M,C,n):

    mu_frontier = np.linspace(0,0.25,1000)
    mu_frontier_no_short = []
    sigma_frontier_no_short = []

    for mu in mu_frontier:
        sigma, w = minVariance(mu,M,C,n)
        if noShort(w):
            sigma_frontier_no_short.append(sigma)
            mu_frontier_no_short.append(mu)

    return mu_frontier_no_short, sigma_frontier_no_short  

def calcWeights(M,C,n,ind):
    
    mu_frontier = np.linspace(0,0.25,1000)
    w1_MVL = []
    w2_MVL = []

    for mu in mu_frontier:
        sigma, w = minVariance(mu,M,C,n)
        if n == 3 or ind == 2:
            w1_MVL.append(w[0])
            w2_MVL.append(w[1])
        elif ind == 0:
            w1_MVL.append(1-w[0]-w[1])
            w2_MVL.append(w[0])
        else:
            w1_MVL.append(w[0])
            w2_MVL.append(1-w[0]-w[1])

    return w1_MVL, w2_MVL  

def calcWeightsNoShort(M,C,n,ind):
    
    mu_frontier = np.linspace(0,0.25,1000)
    w1_MVL_no_short = []
    w2_MVL_no_short = []

    for mu in mu_frontier:
        sigma, w = minVariance(mu,M,C,n)
        if noShort(w):
            if n == 3 or ind == 2:
                w1_MVL_no_short.append(w[0])
                w2_MVL_no_short.append(w[1])
            elif ind == 0:
                w1_MVL_no_short.append(1-w[0]-w[1])
                w2_MVL_no_short.append(w[0])
            else:
                w1_MVL_no_short.append(w[0])
                w2_MVL_no_short.append(1-w[0]-w[1])

    return w1_MVL_no_short,w2_MVL_no_short  


def minVarianceCurve(M,C):

    M12, C12 = getMC(M,C,2)
    M23, C23 = getMC(M,C,0)
    M13, C13 = getMC(M,C,1)

    mu_frontier, sigma_frontier = calcFrontier(M,C,3)
    plt.plot(sigma_frontier,mu_frontier,color = 'red', label = 'All Stocks')
    
    mu_frontier, sigma_frontier = calcFrontier(M12,C12,2)
    plt.plot(sigma_frontier,mu_frontier,color = 'blue', label = 'Stocks 1,2')
    mu_frontier, sigma_frontier = calcFrontier(M23,C23,2)
    plt.plot(sigma_frontier,mu_frontier,color = 'green', label = 'Stocks 2,3')
    mu_frontier, sigma_frontier = calcFrontier(M13,C13,2)
    plt.plot(sigma_frontier,mu_frontier,color = 'purple', label = 'Stocks 1,3')
    
    plt.xlabel('sigma (Risk)') 
    plt.ylabel('mu (Return)')
    plt.title("Minimum Variance Curves and Markowitz Efficient Frontier")
    plt.legend(loc = 'upper left')
    plt.show()
    plt.clf()   

def minVarianceCurveNoShort(M,C):

    M12, C12 = getMC(M,C,2)
    M23, C23 = getMC(M,C,0)
    M13, C13 = getMC(M,C,1)

    mu_frontier, sigma_frontier = calcFrontierNoShort(M,C,3)
    plt.plot(sigma_frontier,mu_frontier,color = 'red', label = 'All Stocks')
    
    mu_frontier, sigma_frontier = calcFrontierNoShort(M12,C12,2)
    plt.plot(sigma_frontier,mu_frontier,color = 'blue', label = 'Stocks 1,2')
    mu_frontier, sigma_frontier = calcFrontierNoShort(M23,C23,2)
    plt.plot(sigma_frontier,mu_frontier,color = 'green', label = 'Stocks 2,3')
    mu_frontier, sigma_frontier = calcFrontierNoShort(M13,C13,2)
    plt.plot(sigma_frontier,mu_frontier,color = 'purple', label = 'Stocks 1,3')
    
    plt.xlabel('sigma (Risk)') 
    plt.ylabel('mu (Return)')
    plt.title("Feasible Set of Portfolios without Short-Selling")
    plt.legend(loc = 'upper left')
    plt.show()
    plt.clf()

def plotWeights(M,C):

    M12, C12 = getMC(M,C,2)
    M23, C23 = getMC(M,C,0)
    M13, C13 = getMC(M,C,1)

    w1,w2 = calcWeights(M,C,3,-1)
    plt.plot(w1,w2,color = 'red', linewidth = 3, label = 'All Stocks')


    w1,w2 = calcWeights(M12,C12,2,2)
    w1_ns,w2_ns = calcWeightsNoShort(M12,C12,2,2)
    plt.plot(w1,w2,color = 'green', linewidth = 1, label = 'Stocks 1,2')
    plt.plot(w1_ns,w2_ns,color = 'green', linewidth = 3)
    
    w1,w2 = calcWeights(M23,C23,2,0)
    w1_ns,w2_ns = calcWeightsNoShort(M23,C23,2,0)
    plt.plot(w1,w2,color = 'blue', linewidth = 1, label = 'Stocks 2,3')
    plt.plot(w1_ns,w2_ns,color = 'blue', linewidth = 3)

    w1,w2 = calcWeights(M13,C13,2,1)
    w1_ns,w2_ns = calcWeightsNoShort(M13,C13,2,1)
    plt.plot(w1,w2,color = 'purple', linewidth = 1, label = 'Stocks 1,3')
    plt.plot(w1_ns,w2_ns,color = 'purple', linewidth = 3)

    plt.axis((-0.25,1.25,-0.25,1.25))
    plt.title("w1 vs w2 (Highlighted represents No Short Selling)")
    plt.xlabel("w1")
    plt.ylabel("w2")
    plt.legend(loc = 'best')
    plt.show()
    plt.clf()

def main():
    M = [0.1, 0.2, 0.15]
    C = [
         [0.005, -0.010, 0.004],
         [-0.010, 0.040, -0.002],
         [0.004, -0.002, 0.023]   
        ]

    M = np.array(M)
    C = np.array(C)

    minVarianceCurve(M,C)
    minVarianceCurveNoShort(M,C)
    plotWeights(M,C)

if __name__ == '__main__':
    main()

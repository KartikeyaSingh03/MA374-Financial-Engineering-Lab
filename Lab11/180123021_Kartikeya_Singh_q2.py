from numpy import exp, log, sqrt, linspace
import matplotlib.pyplot as plt


Beta = [0.02, 0.7, 0.06]
Mu = [0.7, 0.1, 0.09]
Sig = [0.02, 0.3, 0.5]
R0 = [0.1, 0.2, 0.02]

def CIR(beta, mu, sig, r0, T, t):
    a = beta
    b = mu 
    h = sqrt(a**2 + 2*(sig**2))

    A = ((2*h*exp((a+h)*(T-t)/2))/(2*h + (a+h)*(exp((T-t)*h)-1)))**((2*a*b)/(sig**2))

    B = (2*(exp((T-t)*h)-1))/(2*h + (a+h)*(exp((T-t)*h)-1))

    p = A*exp(-r0*B)
    y = -log(p)/(T-t)

    return y

def main():

    T = linspace(1, 10, 10)
     
    for i in range(3):
        beta = Beta[i]
        mu = Mu[i]
        sig = Sig[i]
        r0 = R0[i]

        Y = [CIR(beta, mu, sig, r0, T1,  0) for T1 in T]

        plt.plot(T, Y)
        plt.xlabel("Time")
        plt.ylabel("Yield")
        plt.title("Term structure for [%.2f , %.2f, %.2f, %.1f] (CIR)"%(beta, mu, sig, r0))
        plt.savefig("q2_10_%d"%i)
        plt.clf()

    T = linspace(1, 600, 600)
    R0s = linspace(0.1, 1, 10)

    
    beta = 0.02
    mu = 0.7
    sig = 0.02

    for r0 in R0s:
        Y = [CIR(beta, mu, sig, r0, T1,  0) for T1 in T]
        plt.plot(T, Y, label = 'r = %.1f'%r0)

    plt.xlabel("Time")
    plt.ylabel("Yield")
    plt.title("Yield Curves vs Maturity for [beta, mu, sig] = [%.2f , %.1f, %.2f] (CIR)"%(beta, mu, sig))
    plt.legend(loc = 'best')
    plt.savefig("q2_600")
    plt.clf()

if __name__  == '__main__':
    main()
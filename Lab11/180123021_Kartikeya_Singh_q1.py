from numpy import exp, log, linspace
import matplotlib.pyplot as plt

Beta = [5.9, 3.9, 0.1]
Mu = [0.2, 0.1, 0.4]
Sig = [0.3, 0.3, 0.11]
R0 = [0.1, 0.2, 0.1]

def Vasicek(beta, mu, sig, r0, T, t):
    b = beta*mu
    a = beta

    B = (1/a)*(1 - exp(-a*(T-t)))
    A = (B-T+t)*(a*b - 0.5*sig*sig)/(a*a) - (sig*sig*B*B)/(4*a)

    p = exp(A - B*r0)

    y = -log(p)/(T-t)

    return y

def main():

    T = linspace(1, 10, 10)
     
    for i in range(3):
        beta = Beta[i]
        mu = Mu[i]
        sig = Sig[i]
        r0 = R0[i]

        Y = [Vasicek(beta, mu, sig, r0, T1,  0) for T1 in T]

        plt.plot(T, Y)
        plt.xlabel("Time")
        plt.ylabel("Yield")
        plt.title("Term structure for [%.1f , %.1f, %.2f, %.1f] (Vasicek)"%(beta, mu, sig, r0))
        plt.savefig("q1_10_%d"%i)
        plt.clf()

    T = linspace(1/252, 500/252, 500)
    R0s = linspace(0.1, 1, 10)

    for i in range(3):
        beta = Beta[i]
        mu = Mu[i]
        sig = Sig[i]

        for r0 in R0s:
            Y = [Vasicek(beta, mu, sig, r0, T1,  0) for T1 in T]
            plt.plot(T, Y, label = 'r = %.1f'%r0)

        plt.xlabel("Time")
        plt.ylabel("Yield")
        plt.title("Yield Curves vs Maturity for [beta, mu, sig] = [%.1f , %.1f, %.2f] (Vasicek)"%(beta, mu, sig))
        plt.legend(loc = 'best')
        plt.savefig("q1_500_%d"%i)
        plt.clf()

if __name__  == '__main__':
    main()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate as integ
import scipy.optimize as op
from scipy import stats
from scipy.integrate import odeint

url = 'tunisian_data.csv'
data = pd.read_csv(url)
XX = np.arange(0, len(data), 1)
YY = data.iloc[0:, 2]
ZZ = data.iloc[0:, 3]
Dates = data.iloc[0:, 0]


def model(t, x1, x2, x3):
    return x1 * np.exp(x2 * t) - x3


def estimateParams():
    best_vals, covar = op.curve_fit(model, X, Y)
    x1 = best_vals[0]
    x2 = best_vals[1]
    x3 = best_vals[2]

    return [x1, x2, x3]


def LinReg():
    x3 = estimateParams()[2]
    LogY = np.log(Y + x3)
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, LogY)
    x1 = np.exp(intercept)
    x2 = slope
    t0 = (np.log(x3) - np.log(x1)) / x2

    return [x1, x2, x3, t0]


def CR_fit(t):
    x1 = LinReg()[0]
    x2 = LinReg()[1]
    x3 = LinReg()[2]
    return x1 * np.exp(x2 * t) - x3


def Simu():
    S0 = 11e6  # Population totale
    f = 0.8  # Fraction des cas symptomatiques détectée
    nu = 1 / 7  # 1/nu est le temps moyen durant lequel un infecté reste presymptomatique
    Eta = 1 / 7  # 1/Eta est le temps moyen durant lequel un infecté reste symptomatique (avant de guérir ou décéder)
    nu1 = f * nu
    nu2 = (1 - f) * nu
    Mu = 3.5 / 100
    x1 = LinReg()[0]
    x2 = LinReg()[1]
    x3 = LinReg()[2]
    t0 = LinReg()[3]
    I0 = x3 * x2 / nu1
    tau = (x2 + nu) / S0 * (Eta + x2) / (nu2 + Eta + x2)
    U0 = nu2 / (Eta + x2) * I0
    R0 = tau * S0 / nu * (1 + nu2 / Eta)

    return [nu1, nu2, tau, I0, U0, R0, Eta, Mu]


def I(t):
    I0 = Simu()[3]
    x2 = LinReg()[1]
    t0 = LinReg()[3]
    return I0 * np.exp(x2 * (t - t0))


def U(t):
    U0 = Simu()[4]
    x2 = LinReg()[1]
    t0 = LinReg()[3]
    return U0 * np.exp(x2 * (t - t0))


def deriv(y, t, taut, nu1, nu2, Eta):
    S, I, R, U, RR, RD = y

    tautt = taut(t)
    dS = -tautt * S * (I + U)
    dI = tautt * S * (I + U) - (nu1 + nu2) * I
    dR = nu1 * I - Eta * R
    dU = nu2 * I - Eta * U
    dRR = (Eta * R + Eta * U) * (1 - Simu()[7])
    dRD = (Eta * R + Eta * U) * Simu()[7]

    return dS, dI, dR, dU, dRR, dRD


def SolveSIRU():
    # Conditions initiales
    y0 = 11e6, Simu()[3], 0, Simu()[4], YY[J], ZZ[J]

    # Interval d'intégration de l'ODE
    t = np.asarray(XX)

    # Paramètres du modèle
    nu1 = Simu()[0]
    nu2 = Simu()[1]
    Eta = Simu()[6]

    # Résolution des équations differentielles
    sol = odeint(deriv, y0, t, args=(taut, nu1, nu2, Eta))
    S, I, R, U, RR, RD = sol.T

    CRR = nu1 * integ.cumtrapz(I, initial=t[0]) + YY[J]
    # plt.plot(t, S, label="S")
    plt.plot(t, I, label="I: Infected")
    plt.plot(t, CRR, label="CR: Total Detected")
    plt.plot(t, R, label="R: Detected in quarantine")
    plt.plot(t, U, label="U: Non Detected")
    plt.plot(t, RR, label="RR: Recovered")
    plt.plot(t, RD, label="RD: Dead")
    plt.xlabel("temps en jours")
    plt.ylabel("nombre de cas")
    plt.legend()

    # f = open("CRR.txt", "w+") #a+
    # f = open('CRR.txt', 'r')

    A = np.array(CRR)
    A = A.astype(int)
    with open('CRR.txt', 'w+') as file:
        file.seek(0)
        file.write('\n'.join(str(a) for a in A))
        file.close

    plt.show()


def main():
    p = int(input("Quelle est la vague a analyser ?   "))
    global X, XX, Y, YY, J, taut
    X = np.asarray(XX)
    plt.scatter(X, YY)
    plt.scatter(X, ZZ)

    if p == 1:
        J = 0
        X = XX[0:23]
        Y = YY[0:23]
        print(estimateParams(), Simu()[5])
        X = np.asarray(X)
        fitLine1 = CR_fit(np.asarray(XX[0:45]))
        axes = plt.axes()
        plt.plot(np.asarray(XX[0:45]), fitLine1, c='r')
        axes.grid()
        taut = lambda t: Simu()[2] if t < 23 else 0.315 * Simu()[2]

    elif p == 2:
        J = int(input("A quelle jour demarre la tendance de la deuxieme vague ?   "))
        X = XX[J:len(data)]
        Y = YY[J:len(data)]
        print(LinReg(), Simu()[5])
        X = np.asarray(X)
        XX = np.arange(J, len(data) + 520, 1)
        fitLine2 = CR_fit(XX[0:150])
        plt.plot(XX[0:150], fitLine2, c='r', label="Exponential Fit")
        X = X - J
        Y = Y - YY[J]
        taut = lambda t: Simu()[2] if t < 5000 + 15 else 0.315 * Simu()[2]

    SolveSIRU()


main()

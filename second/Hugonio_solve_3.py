import math
import numpy as np
import scipy.optimize as sc
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

# constants
kB = 1.38064853e-23  # J/K
Avogadro = 6.022140857e+23  # N/mol
aVol = 5.2917720859e-11 ** 3
hartree = 13.605693009 * 2  # eV
mass = 1  # kg/mol
rho0 = 2700.  # kg/m3
R = Avogadro * kB
q0 = 1.60217662e-19  #
eV = q0 / kB  # K


# aEnrg    = kB*hartree*eV
# aPres    = aEnrg/aVol

# УрС идеального газа
class IdealGas:
    def __init__(self, gamma):
        self.gamma = gamma

    def E(self, V, T):
        return self.P(V, T) * V / (self.gamma - 1)

    def P(self, V, T):
        return R * T / V

    # максимальное сжатие ид. газа ударной волной
    # Зельдович, Райзер стр. 52
    def maxCompressionRate(self):
        return (self.gamma + 1) / (self.gamma - 1)


class BiTerm:
    def __init__(self, gamma, c0, rho0):
        self.gamma = gamma
        self.c0 = c0
        self.rho0 = rho0

    def E(self, V, T):
        return (self.P(V, T) - self.c0 ** 2 * (1. / V - self.rho0)) / (self.gamma - 1) * V

    def P(self, V, T):
        return R * T / V


class Tait:
    def __init__(self, gamma, rho0, c0, n):
        self.gamma = gamma
        self.rho0 = rho0
        self.c0 = c0
        self.n = n

    def E(self, V, P):
        self.const = self.rho0 * self.c0 ** 2 / self.n
        Pref = self.const * ((1. / self.rho0 / V) ** self.n - 1)
        Eref = self.const / (self.n - 1) * (((1 / self.rho0 / V) ** self.n - 1 + self.n) * V - self.n / self.rho0)
        return Eref + (P - Pref) / self.gamma * V


with open("Aluminum_data_ready.txt") as f:
    title = f.readline()
    data = f.readlines()
data = list(map(lambda x: x.split(), data))
dV_exp = list(map(lambda x: float(x[4]), data))
P_exp = list(map(lambda x: float(x[3]), data))

ans_s = 100000
ans_gamma = 0
ans_n = 0
# for gamma in np.linspace(0.1, 1, 21):
for gamma in [0.55]:
    # for n in np.linspace(2, 10, 17):
    for n in [3]:
        eos = Tait(rho0=2700, c0=6400, gamma=gamma, n=n)
        # начальная температура
        T0 = 300  # K
        # начальный объем
        V0 = mass / rho0
        # начальное давлениe
        P0 = 1e-4
        # начальная энергия
        E0 = eos.E(V0, P0)
        # диапазон поиска решений уравнения Гюгонио по температуре
        Tmin = 300
        Tmax = 1e+7
        # диапазон поиска решений уравнения Гюгонио по объему
        Vmin = 0.2 * V0
        Vmax = 1.0 * V0


        def hugoniot_V(V, P1):
            result = eos.E(V, P1) - E0 - 0.5 * (P1 + P0) * (V0 - V)  # (1.71)
            return result


        # массив значений давления, для которых ищем решение
        # уравнения Гюгонио
        # P1 = 10 ** np.linspace(np.log10(P0), np.log10(P0) + 19, 30)
        P1 = np.array([P0] + P_exp) * 1e9
        # массив объемов - решений уравнения Гюгонио
        V1 = [V0]
        for Pi in P1[1:]:
            Vi = root_scalar(
                f=hugoniot_V,
                method='newton',
                x0=R * T0 / Pi,
                rtol=1e-6,
                args=(Pi)
            ).root
            # print("P1 = ", Pi, ", V1 = ", Vi)
            V1.append(Vi)

        V1 = V0 / np.array(V1)
        s = 0
        for p in range(1, len(P1)):
            s += (V1[p] - dV_exp[p - 1]) ** 2
        print(f"gamma = {gamma}, n = {n}, s = {(s / len(P1)) ** 0.5}")
        if s and s < ans_s:
            ans_s = s
            ans_gamma = gamma
            ans_n = n
        graph = [(V1[i], P1[i] * 1e-9) for i in range(len(P1))]
        graph.sort(key=lambda x: x[0])
        X = [graph[i][0] for i in range(len(graph))]
        Y = [graph[i][1] for i in range(len(graph))]
        plt.plot(X, Y)
        plt.scatter(dV_exp, P_exp)
        plt.grid()
        plt.xlabel("V0/V1")
        plt.ylabel("P, ГПа")
        plt.show()

# print("Optimal")
# print(f"n = {ans_n}, gamma = {ans_gamma}")

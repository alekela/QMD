import math
import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

# constants
kB = 1.38064853e-23  # J/K
Avogadro = 6.022140857e+23  # N/mol
aVol = 5.2917720859e-11 ** 3
hartree = 13.605693009 * 2  # eV
mass = 27.e-3  # kg/mol
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

    def E(self, V, P):
        return (P - self.c0 ** 2 * (1. / V - self.rho0)) / (self.gamma - 1) * V


# УрС
eos = BiTerm(gamma=2, c0=2.2, rho0=1.)
# начальная температура
T0 = 300  # K
# начальный объем
V0 = mass / rho0
# начальное давлениe
P0 = 1000
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
P1 = 10 ** np.linspace(np.log10(P0), np.log10(P0) + 3, 30)
# массив объемов - решений уравнения Гюгонио
V1 = [V0]
for Pi in P1[1:]:
    Vi = root_scalar(
        f=hugoniot_V,
        method='newton',
        x0=Vmin,
        rtol=1e-6,
        args=(Pi)
    ).root
    print("P1 = ", Pi, ", V1 = ", Vi)
    V1.append(Vi)

V1 = np.array(V1)
P1 *= 1e-9  # ГПа
plt.plot(V0 / V1, P1)
plt.grid()
plt.xlabel("V0/V1")
plt.ylabel("P, ГПа")
plt.show()

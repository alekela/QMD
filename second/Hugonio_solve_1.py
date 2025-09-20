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


# УрС
eos = IdealGas(gamma=5.0 / 3.0)
# начальная температура
T0 = 300  # K 
# начальный объем
V0 = mass / rho0
# начальное давлениe
P0 = eos.P(V0, T0)
# начальная энергия
E0 = eos.E(V0, T0)
# диапазон поиска решений уравнения Гюгонио по температуре
Tmin = 300
Tmax = 1e+7
# диапазон поиска решений уравнения Гюгонио по объему
Vmin = 0.2 * V0
Vmax = 1.0 * V0


# функция для поиска температуры для заданного давления P1
def resolve_T(T, V, P1):
    result = eos.P(V, T) - P1
    return result


# функция для поиска объема - решения уравнения Гюгонио
def hugoniot_V(V, P1):
    T = Tmin
    T = root_scalar(
        f=resolve_T,
        method='newton',
        x0=Tmin,
        rtol=1e-8,
        args=(V, P1)
    ).root
    assert (np.isclose(T, P1 * V / R, atol=1e-8))
    result = eos.E(V, T) - E0 - 0.5 * (P1 + P0) * (V0 - V)  # (1.71)
    return result


# массив значений давления, для которых ищем решение
# уравнения Гюгонио
P1 = 10 ** np.linspace(math.log10(P0), math.log10(P0) + 3, 30)
# массив объемов - решений уравнения Гюгонио
V1 = [V0]
for Pi in P1[1:]:
    Vi = root_scalar(
        f=hugoniot_V,
        method='brentq',
        bracket=(Vmin, Vmax),
        rtol=1e-6,
        args=(Pi,)
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

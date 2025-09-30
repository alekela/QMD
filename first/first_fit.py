import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

with open("Experiment_data_TV.csv") as f:
    title = f.readline()
    data = f.readlines()
data = list(map(lambda x: x.split(',')[:2], data))
Ts = np.array(list(map(lambda x: float(x[0]), data)))
Vs = np.array(list(map(lambda x: float(x[1]), data)))

p = np.polyfit(Ts, Vs, 12)
Vs_fit = np.polyval(p, Ts)


def fit_func(x, p1, p2, p3, p4, p5):
    return np.heaviside(x - p1, 0) * (x * p2 + p3) + (1 - np.heaviside(x - p1, 0)) * (x * p4 + p5)


popt, pcov = curve_fit(fit_func, Ts, Vs)
plt.scatter(Ts, Vs, s=2)
plt.plot(Ts, fit_func(Ts, *popt), color='orange')
plt.plot(Ts, Vs_fit, color='red')

plt.xlabel("T, K")
plt.ylabel("V, cm^3")
plt.grid()
plt.legend(["Actual data", "curve_fit", "Polyfit"])
plt.show()

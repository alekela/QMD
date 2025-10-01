import numpy as np

class IdealGas:

    def __init__(self, gamma):
        self.gamma = gamma

    def pressure(self, density, energy):
        return (self.gamma - 1) * density * energy

    def soundSpeed(self, density, energy):
        return np.sqrt(self.gamma * (self.gamma - 1) * energy)
        
    def energy(self, density, pressure):
        return (pressure / ((self.gamma - 1) * density))

class TaitMurnaghan:
    def __init__(self, rho0, c0, n, gamma):
        self.rho0  = rho0
        self.c0    = c0
        self.n     = n
        self.gamma = gamma

    def pressure(self, density, energy):
        return self.pref(density) + self.gamma*density*(energy - self.eref(density))

    def soundSpeed(self, density, energy):
        gamma = self.gamma
        rho0  = self.rho0
        c0    = self.c0
        B     = rho0 * c0**2
        return np.sqrt(self.gamma * (1 + self.gamma) * (energy - self.eref(density)) + 
               (B + n * self.pref(density))/density)
        
    def energy(self, density, pressure):
        return (pressure - self.pref(density)) / (self.gamma * density) + self.eref(density)

    def pref(self, density):
        rho0  = self.rho0
        c0    = self.c0
        x     = density/self.rho0
        n     = self.n
        B     = rho0 * c0**2
        return B/n * (x**n - 1)

    def eref(self, density):
        rho0  = self.rho0
        c0    = self.c0
        x     = density/self.rho0
        n     = self.n
        result  = 1/(n - 1)*(x**(n - 1) - 1)
        result += 1/x - 1
        result *= c0**2/n
        return result

class Hugoniot:
    def __init__(self, rho0, c0, s, gamma):
        self.rho0  = rho0
        self.c0    = c0
        self.s     = s
        self.gamma = gamma

    def pressure(self, density, energy):
        return self.pref(density) + self.gamma*density*(energy - self.eref(density))

    def soundSpeed(self, density, energy):
        rho0 = self.rho0
        c0   = self.c0
        s    = self.s
        return np.sqrt(c0**2 + 4.0*s*self.pressure(density, energy)/rho0)
        
    def energy(self, density, pressure):
        return (pressure - self.pref(density)) / (self.gamma * density) + self.eref(density)

    def pref(self, density):
        rho0  = self.rho0
        c0    = self.c0
        x     = self.rho0/density
        s     = self.s
        B     = rho0 * c0**2
        tmp   = 1.0 - (1.0 - x)*s;
        result = np.zeros_like(density)
        ixle1 = np.where(x <  1)
        ixge1 = np.where(x >= 1)
        result[ixle1] = B * (1 - x[ixle1]) / (tmp[ixle1]**2)
        result[ixge1] = B * (1 - x[ixge1])
        return result

    def eref(self, density):
        rho0  = self.rho0
        c0    = self.c0
        x     = self.rho0/density
        s     = self.s
        B     = rho0 * c0**2
        result = np.zeros_like(density)
        ixle1 = np.where(x <  1)
        ixge1 = np.where(x >= 1)
        result[ixle1] = 0.5*self.pref(density[ixle1])*(1.0/rho0 - 1.0/density[ixle1]) 
        result[ixge1] = 0.5*B*(1.0 - x[ixge1])**2/rho0
        return result


class HugoniotTwoPhase:
    def __init__(self, rho01,     rho02,
                       rho12from, rho12to,
                       rho21from, rho21to, 
                       c01,       c02, 
                       s1,        s2, 
                       gamma1,    gamma2):

        self.rho01     = rho01;     self.rho02   = rho02;
        self.rho12from = rho12from; self.rho12to = rho12to;
        self.rho21from = rho21from; self.rho21to = rho21to; 
        self.c01       = c01;       self.c02     = c02; 
        self.s1        = s1;        self.s2      = s2; 
        self.gamma1    = gamma1;    self.gamma2  = gamma2;

    def phase(self, density, energy, phase):
        newPhase = np.ones_like(phase)*phase
        newPhase[density <= self.rho21to] = 1
        newPhase[density >= self.rho12to] = 2
        return newPhase

    def pressure(self, density, energy, phase):
        icase1 = np.where((density <  self.rho12from) * (phase == 1))
        icase2 = np.where((density >= self.rho12from) * (density <  self.rho12to)   * (phase == 1))
        icase3 = np.where((density >= self.rho21from) * (phase == 2))
        icase4 = np.where((density >= self.rho21to)   * (density <  self.rho21from) * (phase == 2))

        P = np.zeros_like(density)
        P[icase1] = self.pref1(density[icase1]) + self.gamma1*density[icase1]*(energy[icase1] - self.eref1(density[icase1]))
        P[icase2] = self.pref1(density[icase2]/density[icase2]*self.rho12from) + self.gamma1*density[icase2]*(energy[icase2] - self.eref1(density[icase2]/density[icase2]*self.rho12from))
        P[icase3] = self.pref2(density[icase3]) + self.gamma2*density[icase3]*(energy[icase3] - self.eref2(density[icase3]))
        P[icase4] = self.pref2(density[icase4]/density[icase4]*self.rho21from) + self.gamma2*density[icase4]*(energy[icase4] - self.eref2(density[icase4]/density[icase4]*self.rho21to))

        return P

    def soundSpeed(self, density, energy, phase):
        icase1 = np.where((density <  self.rho12from) * (phase == 1))
        icase2 = np.where((density >= self.rho12from) * (density <  self.rho12to)   * (phase == 1))
        icase3 = np.where((density >= self.rho21from) * (phase == 2))
        icase4 = np.where((density >= self.rho21to)   * (density <  self.rho21from) * (phase == 2))

        c = np.zeros_like(density)
        P = self.pressure(density, energy, phase)
        c[icase1] = self.rho01/density[icase1] * np.sqrt(self.c01**2 + 4.0*self.s1*P[icase1]/self.rho01)
        c[icase2] = self.rho01/self.rho12from  * np.sqrt(self.c01**2 + 4.0*self.s1*P[icase2]/self.rho01)
        c[icase3] = self.rho02/density[icase3] * np.sqrt(self.c02**2 + 4.0*self.s2*P[icase3]/self.rho02)
        c[icase4] = self.rho02/self.rho21from  * np.sqrt(self.c02**2 + 4.0*self.s2*P[icase4]/self.rho02)
        
        return c
        
    def energy(self, density, pressure):
        return 0 # для задачи не нужно

    def pref1(self, density):
        rho0  = self.rho01
        c0    = self.c01
        x     = self.rho01/density
        s     = self.s1
        B     = rho0 * c0**2
        tmp   = 1.0 - (1.0 - x)*s;
        result = np.zeros_like(density)
        ixle1 = np.where(x <  1)
        ixge1 = np.where(x >= 1)
        result[ixle1] = B * (1 - x[ixle1]) / (tmp[ixle1]**2)
        result[ixge1] = B * (1 - x[ixge1])
        return result

    def eref1(self, density):
        rho0  = self.rho01
        c0    = self.c01
        x     = self.rho01/density
        s     = self.s1
        B     = rho0 * c0**2
        result = np.zeros_like(density)
        ixle1 = np.where(x <  1)
        ixge1 = np.where(x >= 1)
        result[ixle1] = 0.5*self.pref1(density[ixle1])*(1.0/rho0 - 1.0/density[ixle1]) 
        result[ixge1] = 0.5*B*(1.0 - x[ixge1])**2/rho0
        return result

    def pref2(self, density):
        rho0  = self.rho02
        c0    = self.c02
        x     = self.rho02/density
        s     = self.s2
        B     = rho0 * c0**2
        tmp   = 1.0 - (1.0 - x)*s;
        result = np.zeros_like(density)
        ixle1 = np.where(x <  1)
        ixge1 = np.where(x >= 1)
        result[ixle1] = B * (1 - x[ixle1]) / (tmp[ixle1]**2)
        result[ixge1] = B * (1 - x[ixge1])
        return result

    def eref2(self, density):
        rho0  = self.rho02
        c0    = self.c02
        x     = self.rho02/density
        s     = self.s2
        B     = rho0 * c0**2
        result = np.zeros_like(density)
        ixle1 = np.where(x <  1)
        ixge1 = np.where(x >= 1)
        result[ixle1] = 0.5*self.pref2(density[ixle1])*(1.0/rho0 - 1.0/density[ixle1]) 
        result[ixge1] = 0.5*B*(1.0 - x[ixge1])**2/rho0
        return result

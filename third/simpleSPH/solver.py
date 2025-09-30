import numpy as np
from .data import field, Storage

class WendlandC2:

    def __call__(self, r, h):

        q = r / h
        ind = np.where(q < 1)
        w = np.zeros_like(q)
        w[ind] = ((5. / 4. * (1. + 3 * q) * (1. - q) ** 3) / h)[ind]
        
        return w

    def derivative(self, r, h):

        q = r / h
        ind = np.where(q < 1)
        w = np.zeros_like(q)
        w[ind] = ((-15. * q * (-q + 1.) **2) / (h ** 2))[ind]
        
        return w

class RiemannSolver:

    def velocity(self, particleL, particleR):
        # значения плотности слева и справа от контакта
        rL = particleL[field.density]
        rR = particleR[field.density]
        # значения скорости слева и справа от контакта
        vL = particleL[field.velocity]
        vR = particleR[field.velocity]
        # значения скорости звука слева и справа от контакта
        cL = particleL[field.soundSpeed]
        cR = particleR[field.soundSpeed]
        # значения давления слева и справа от контакта
        pL = particleL[field.pressure]
        pR = particleR[field.pressure]

        zL = rL*cL
        zR = rR*cR

        vC = zL*vL + zR*vR - (pR - pL)  
        vC /= zL + zR

        return vC - vL 

    def pressure(self, particleL, particleR):
        # значения плотности слева и справа от контакта
        rL = particleL[field.density]
        rR = particleR[field.density]
        # значения скорости слева и справа от контакта
        vL = particleL[field.velocity]
        vR = particleR[field.velocity]
        # значения скорости звука слева и справа от контакта
        cL = particleL[field.soundSpeed]
        cR = particleR[field.soundSpeed]
        # значения давления слева и справа от контакта
        pL = particleL[field.pressure]
        pR = particleR[field.pressure]

        zL = rL*cL
        zR = rR*cR

        pC = (zL*pL + zR*pR - zL*zR*(vR - vL))/(zL + zR)

        return pC

    def energy(self, particleL, particleR):

        vL = particleL[field.velocity]
        vC = self.velocity(particleL, particleR)
        pC = self.pressure(particleL, particleR)

        return (vC + vL)*pC

class SPHSolver:
    def __init__(self, kernel = WendlandC2(), 
                       contact = RiemannSolver(), 
                       smoothing_scale = 1.0, 
                       period = 1e+15
        ):
        # kernel - функция сглаживающего ядра, по-умолчанию - WendlandC2
        self.kernel = kernel
        # contact - функция расчета контактного взаимодействия
        self.contact = contact
        # smoothing_scale - модификатор длины сглаживания
        self.smoothing_scale = smoothing_scale
        # period - длина расчетной области для периодических граничных условий
        self.period = period

        self.neighbors = Storage()
    
    @property
    def derivatives(self):
        # производные, которые вычисляет решатель
        return {
            field.coords:   field.velocity,
            field.velocity: field.force,
            field.density:  field.densityDerivative,
            field.energy:   field.energyDerivative
        }

    def __call__(self, particles):
        # производная сглаживающего ядра
        dW = self.kernel.derivative

        # расчет
        if self.neighbors.size != particles.size:
            self.neighbors = Storage(fields=particles.fields, size=particles.size)

        neighbors = self.neighbors

        strainRate       = np.zeros(particles.size)
        force            = np.zeros(particles.size)
        energyDerivative = np.zeros(particles.size)

        for ineib in range(0, particles[field.neibs][0,:].size):
            for f in particles.fields:
            	if f != field.neibs:
                	neighbors[f][:] = particles[f][particles[field.neibs][:,ineib]]

            r_ji = neighbors[field.coords] - particles[field.coords]
            iprd1 = np.where(r_ji > 0.5*self.period)
            iprd2 = np.where(r_ji < -0.5*self.period)
            r_ji[iprd1] -= self.period
            r_ji[iprd2] += self.period

            r = np.abs(r_ji)
            e_ji = r_ji/r
            h = self.smoothing_scale*(particles[field.size] + neighbors[field.size])

            particles[field.velocity] *= e_ji
            neighbors[field.velocity] *= e_ji

            D = neighbors[field.size]

            strainRate       += D * self.contact.velocity(particles, neighbors) * dW(r, h)
            force            += D * self.contact.pressure(particles, neighbors) * dW(r, h) * e_ji # вклад от частицы j
            energyDerivative += D * self.contact.energy(particles, neighbors) * dW(r, h) # вклад от частицы j

            particles[field.velocity] *= e_ji
            neighbors[field.velocity] *= e_ji

        particles[field.strainRate]        = 2 * strainRate
        particles[field.force]             = 2 * force/particles[field.density]
        particles[field.energyDerivative]  = 2 * energyDerivative/particles[field.density]
        particles[field.densityDerivative] = 2 * strainRate * particles[field.density]

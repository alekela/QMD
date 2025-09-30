import math
import numpy as np

from simpleSPH.data import field, Storage
from simpleSPH import particles
from simpleSPH.eos import Hugoniot
from simpleSPH.solver import WendlandC2, RiemannSolver, SPHSolver as Solver
from simpleSPH.stepper import RK1 as Stepper

# Параметры задачи impact 
nelem            = 400   # число ячеек
LImpactor        = 5     # мм
LTarget          = 5     # мм
materialImpactor = 1     # id материала
materialTarget   = 2     # id материала
RhoImpactor      = 2700  # плотность ударника
RhoTarget        = 7900  # плотность мишени
С0Impactor       = 5380  # скорость звука ударник
С0Target         = 4570  # скорость звука мишень
GrImpactor       = 2.0   # параметр Грюнайзена ударника
GrTarget         = 2.0   # параметр Грюнайзена мишени
sImpactor        = 1.34  # параметр ударной адиабаты ударника
sTarget          = 1.49  # параметр ударной адиабаты мишени
vImpactor        = 1000  # скорость ударника м/c
tend             = 0.003  # время окончания моделирования мс

impactorEos = Hugoniot(
	rho0  = RhoImpactor, 
	c0    = С0Impactor, 
	s     = sImpactor, 
	gamma = GrImpactor
)

targetEos = Hugoniot(
	rho0  = RhoTarget, 
	c0    = С0Target, 
	s     = sTarget, 
	gamma = GrTarget
)

# начальное распределение плотности 
def density0(x):
	rho = np.zeros_like(x)
	rho[x <  0] = RhoImpactor
	rho[x >= 0] = RhoTarget
	return rho

def material0(x):
	material = np.zeros_like(x)
	material[x <  0] = materialImpactor
	material[x >= 0] = materialTarget
	return material

# начальное распределение скорости
def velocity0(x):
	v = np.zeros_like(x)
	v[x <  0] = vImpactor
	v[x >= 0] = 0.0
	return v

# сетка частиц
elements = particles.create(
	xmin = -LImpactor,
	xmax =  LTarget,
	n    =  nelem,
	fields = [ 
		field.mass,
		field.coords,
		field.velocity,
		field.force,
		field.density,
		field.strainRate,
		field.densityDerivative,
		field.pressure,
		field.energy,
		field.energyDerivative,
		field.innerEnergy,
		field.soundSpeed,
		field.size,
		field.neibs,
		field.material
	],
	nneibs = 6 # число соседних частиц в списке
)
# задание начальных полей на сетке
# записать в ячейки начальные значения плотности, скорости
# внутренней энергии и массы частиц
elements[field.density]     = density0(elements[field.coords])
elements[field.velocity]    = velocity0(elements[field.coords])
elements[field.innerEnergy] = np.zeros_like(elements[field.density])
elements[field.material]    = material0(elements[field.coords])
elements[field.mass]        = elements[field.density]*elements[field.size]
# функция применяется перед интегрированием - расчет полной энергии
def before_step(particles):
	particles[field.energy] = particles[field.innerEnergy] + 0.5*particles[field.velocity]**2

# функция применяется после интегрирования:
# расчет внутренней энергии и обновление размера частиц
def after_step(particles):
	particles[field.innerEnergy] = particles[field.energy] - 0.5*particles[field.velocity]**2
	particles[field.size]        = particles[field.mass]/particles[field.density]

# расчет параметров УрС
def model(particles):
	iImpactor = np.where(particles[field.material] == materialImpactor)
	iTarget   = np.where(particles[field.material] == materialTarget)
	particles[field.pressure][iImpactor]   = impactorEos.pressure(particles[field.density][iImpactor], particles[field.innerEnergy][iImpactor])
	particles[field.soundSpeed][iImpactor] = impactorEos.soundSpeed(particles[field.density][iImpactor], particles[field.innerEnergy][iImpactor])
	particles[field.pressure][iTarget]     = targetEos.pressure(particles[field.density][iTarget], particles[field.innerEnergy][iTarget])
	particles[field.soundSpeed][iTarget]   = targetEos.soundSpeed(particles[field.density][iTarget], particles[field.innerEnergy][iTarget])

# SPH решатель, выбираем тип сглаживающего ядра и 
# метод расчета значений на контактном разрыве
solver = Solver(contact=RiemannSolver(), kernel=WendlandC2(), smoothing_scale=0.93)
# Интегрирование по времени по методу Эйлера
stepper = Stepper(
	solver     = solver, # решатель для расчета правой части
	models     = model, # применение моделей материалов
	derivative = solver.derivatives, # список величин для интегрирования
	CFL        = 0.5, # число Courant-Friedrichs-Levi
	before     = before_step, # перед шагом по времени
	after      = after_step  # после шага по времени
)
# расчет начального давления и скорости звука
time  = 0.0
istep = 0
visar = []
model(elements)
# генератор шага интегрирования
step = stepper.update(elements)
# цикл шагов по времени
while time < tend: 
	# используем генератор step
	time = next(step)
	istep += 1
	if istep % 100 == 0:
		print(istep, time)
	visar.append([time, elements[field.velocity][-1]])

visar = np.array(visar)
tvisar = visar[:,0]
vvisar = visar[:,1]

import matplotlib.pyplot as plt

# строим график для плотности
# plt.plot(elements.data[field.coords], elements.data[field.velocity])
# plt.xlim(-LImpactor, 1.5*LTarget)

plt.plot(tvisar, vvisar)
plt.show()

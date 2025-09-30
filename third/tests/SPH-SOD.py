import math
import numpy as np

from simpleSPH.data import field, Storage
from simpleSPH import particles
from simpleSPH.eos import IdealGas
from simpleSPH.solver import WendlandC2, RiemannSolver, SPHSolver as Solver
from simpleSPH.stepper import RK1 as Stepper

import sodexact

# Параметры задачи SOD 
gamma = 3.0  # показатель адиабаты ид. газа
nelem = 400  # число ячеек
densL = 0.15  # плотность слева от разрыва
densR = 0.12  # плотность справа от разрыва
pL = 3.0  # давление слева от разрыва
pR = 1.0  # давление справа от разрыва
vL = 0.0  # скорость слева от разрыва
vR = 0.0  # скорость справа от разрыва
tend = 0.02  # время окончания моделирования

# модель УрС идеального газа
gas = IdealGas(gamma)


# начальное распределение плотности 
def density0(x):
    # слева от 0 densL, справа densR
    if x < 0:
        return densL
    else:
        return densR


# начальное распределение скорости
def velocity0(x):
    # слева от 0 vL, справа vR
    if x < 0:
        return vL
    else:
        return vR


# начальное распределение внутренней энергии (давления)
def innerEnergy0(x):
    # слева от 0 давление должно быть pL, справа pR
    # для расчета энергии использовать УрС gas
    if x < 0:
        return gas.energy(densL, pL)
    else:
        return gas.energy(densR, pR)


# сетка частиц
elements = particles.create(
    xmin=-0.5,
    xmax=0.5,
    n=nelem,
    fields=[
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
        field.neibs
    ],
    nneibs=10  # число соседних частиц в списке
)
# задание начальных полей на сетке
for i in range(nelem):
    # записать в ячейки начальные значения плотности, скорости
    # внутренней энергии и массы частиц
    elements[field.density][i] = density0(elements[field.coords][i])
    elements[field.velocity][i] = velocity0(elements[field.coords][i])

    elements[field.innerEnergy][i] = innerEnergy0(elements[field.coords][i])
    elements[field.mass][i] = elements[field.density][i] * elements[field.size][i]


# функция применяется перед интегрированием - расчет полной энергии
def before_step(particles):
    for i in range(particles.size):
        particles[field.energy][i] = particles[field.innerEnergy][i] + 0.5 * particles[field.velocity][i] ** 2


# функция применяется после интегрирования:
# расчет внутренней энергии и обновление размера частиц
def after_step(particles):
    for i in range(particles.size):
        particles[field.innerEnergy][i] = particles[field.energy][i] - 0.5 * particles[field.velocity][i] ** 2
        particles[field.size][i] = particles[field.mass][i] / particles[field.density][i]


# для расчета давления и скорости звука использовать
# УрС идеального газа gas
def model(cells):
    for i in range(cells.size):
        cells[field.pressure][i] = gas.pressure(cells[field.density][i], cells[field.innerEnergy][i])
        cells[field.soundSpeed][i] = gas.soundSpeed(cells[field.density][i], cells[field.innerEnergy][i])


# SPH решатель, выбираем тип сглаживающего ядра и
# метод расчета значений на контактном разрыве
solver = Solver(contact=RiemannSolver(), kernel=WendlandC2(), smoothing_scale=0.93)
# Интегрирование по времени по методу Эйлера
stepper = Stepper(
    solver=solver,  # решатель для расчета правой части
    models=model,  # применение моделей материалов
    derivative=solver.derivatives,  # список величин для интегрирования
    CFL=0.5,  # число Courant-Friedrichs-Levi
    before=before_step,  # перед шагом по времени
    after=after_step  # после шага по времени
)
# расчет начального давления и скорости звука
model(elements)

time = 0.0
# генератор шага интегрирования
step = stepper.update(elements)
# цикл шагов по времени
while time < tend:
    # используем генератор step
    time = next(step)
    # print(time)

# точное (аналитическое) решение задачи SOD
solution = sodexact.solution(gamma, densL, vL, pL, densR, vR, pR, tend, nelem)
# дополняем ячейки полями с точным решением
elements.data[field.coordsSolution] = solution[field.coords]
elements.data[field.densitySolution] = solution[field.density]
elements.data[field.velocitySolution] = solution[field.velocity]
elements.data[field.pressureSolution] = solution[field.pressure]
elements.data[field.energySolution] = solution[field.energy]

import matplotlib.pyplot as plt

# строим график для плотности
plt.plot(elements.data[field.coords], elements.data[field.density])
plt.plot(elements.data[field.coordsSolution], elements.data[field.densitySolution])
plt.xlim(-0.25, 0.25)
plt.ylim(0.1, 0.16)
plt.show()

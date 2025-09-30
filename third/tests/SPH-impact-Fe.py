import math
import numpy as np

from simpleSPH.data import field, Storage
from simpleSPH import particles
from simpleSPH.eos import Hugoniot
from simpleSPH.solver import WendlandC2, RiemannSolver, SPHSolver as Solver
from simpleSPH.stepper import RK1 as Stepper
from scipy.optimize import minimize


def main(c0, Gr, s):
    # Параметры задачи impact
    nelem = 1200  # число ячеек
    LImpactor = 6.3  # мм
    LTarget = 6.3  # мм
    materialImpactor = 1  # id материала
    materialTarget = 2  # id материала
    RhoImpactor = 7900  # плотность ударника
    RhoTarget = 7900  # плотность мишени
    С0Impactor = c0  # скорость звука ударник
    С0Target = c0  # скорость звука мишень
    GrImpactor = Gr  # параметр Грюнайзена ударника
    GrTarget = Gr  # параметр Грюнайзена мишени
    sImpactor = s  # параметр ударной адиабаты ударника
    sTarget = s  # параметр ударной адиабаты мишени
    vImpactor = 671  # скорость ударника м/c
    tend = 4e-3  # время окончания моделирования мс

    impactorEos = Hugoniot(
        rho0=RhoImpactor,
        c0=С0Impactor,
        s=sImpactor,
        gamma=GrImpactor
    )
    targetEos = Hugoniot(
        rho0=RhoTarget,
        c0=С0Target,
        s=sTarget,
        gamma=GrTarget
    )

    # начальное распределение плотности
    def density0(x):
        rho = np.zeros_like(x)
        rho[x < 0] = RhoImpactor
        rho[x >= 0] = RhoTarget
        return rho

    def material0(x):
        material = np.zeros_like(x)
        material[x < 0] = materialImpactor
        material[x >= 0] = materialTarget
        return material

    # начальное распределение скорости
    def velocity0(x):
        v = np.zeros_like(x)
        v[x < 0] = vImpactor
        v[x >= 0] = 0.0
        return v

    # сетка частиц
    elements = particles.create(
        xmin=-LImpactor,
        xmax=LTarget,
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
            field.neibs,
            field.material
        ],
        nneibs=6  # число соседних частиц в списке
    )
    # задание начальных полей на сетке
    # записать в ячейки начальные значения плотности, скорости
    # внутренней энергии и массы частиц
    elements[field.density] = density0(elements[field.coords])
    elements[field.velocity] = velocity0(elements[field.coords])
    elements[field.innerEnergy] = np.zeros_like(elements[field.density])
    elements[field.material] = material0(elements[field.coords])
    elements[field.mass] = elements[field.density] * elements[field.size]

    # функция применяется перед интегрированием - расчет полной энергии
    def before_step(particles):
        particles[field.energy] = particles[field.innerEnergy] + 0.5 * particles[field.velocity] ** 2

    # функция применяется после интегрирования:
    # расчет внутренней энергии и обновление размера частиц
    def after_step(particles):
        particles[field.innerEnergy] = particles[field.energy] - 0.5 * particles[field.velocity] ** 2
        particles[field.size] = particles[field.mass] / particles[field.density]

    # расчет параметров УрС
    def model(particles):
        iImpactor = np.where(particles[field.material] == materialImpactor)
        iTarget = np.where(particles[field.material] == materialTarget)
        particles[field.pressure][iImpactor] = impactorEos.pressure(particles[field.density][iImpactor],
                                                                    particles[field.innerEnergy][iImpactor])
        particles[field.soundSpeed][iImpactor] = impactorEos.soundSpeed(particles[field.density][iImpactor],
                                                                        particles[field.innerEnergy][iImpactor])
        particles[field.pressure][iTarget] = targetEos.pressure(particles[field.density][iTarget],
                                                                particles[field.innerEnergy][iTarget])
        particles[field.soundSpeed][iTarget] = targetEos.soundSpeed(particles[field.density][iTarget],
                                                                    particles[field.innerEnergy][iTarget])

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
    time = 0.0
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
        # if istep % 100 == 0:
        #    print(istep, time)
        visar.append([time, elements[field.velocity][-1]])

    visar = np.array(visar)
    return visar


def loss_func(x_exp, y_exp, x, y):
    loss = []
    max_chisl = -1000
    for i in range(len(x_exp)):
        y_chisl = -1000
        for j in range(len(x)):
            if x[j] > x_exp[i]:
                if j != 0:
                    y_chisl = (y[j] - y[j - 1]) / (x[j] - x[j - 1]) * (x_exp[i] - x[j - 1]) + y[j - 1]
                break
        if abs(y_chisl) > max_chisl:
            max_chisl = abs(y_chisl)
        loss.append(abs(y_exp[i] - y_chisl))
    return sum(loss) / max_chisl


import matplotlib.pyplot as plt

with open("Experiment_data_1_phase.csv") as f:
    title = f.readline()
    data = f.readlines()
data = list(map(lambda x: x.split(','), data[5:]))
ts_exp = list(map(lambda x: float(x[0]) * 1e-3, data))
vs_exp = list(map(lambda x: float(x[1]) * 1e3, data))

min_loss = 100000000
min_p = 0


def fun(x):
    global iters
    c0 = x[0]
    Gr = x[1]
    s = x[2]
    visar = main(c0, Gr, s)
    tvisar = visar[:, 0]
    vvisar = visar[:, 1]
    loss = loss_func(ts_exp, vs_exp, tvisar, vvisar)
    iters += 1
    print("Smth")
    return loss


iters = 0
c0 = 4500
Gr = 2
s = 1.1
p_opt = minimize(fun, [c0, Gr, s], method='Nelder-Mead')
print(p_opt)
# for p in np.linspace(1.5, 2.5, 101):
visar = main(p_opt.x[0], p_opt.x[1], p_opt.x[2])
tvisar = visar[:, 0]
vvisar = visar[:, 1]
loss = loss_func(ts_exp, vs_exp, tvisar, vvisar)

print(f"Found optimal c0 = {p_opt.x[0]}, Gr = {p_opt.x[1]}, s = {p_opt.x[2]} with loss {loss}")
print(f"Iters = {iters}")
# строим график для плотности
# plt.plot(elements.data[field.coords], elements.data[field.velocity])
# plt.xlim(-LImpactor, 1.5*LTarget)

plt.plot(tvisar, vvisar)
plt.scatter(ts_exp, vs_exp)
plt.xlabel("t, s* 1e-3")
plt.ylabel("v, m/s")
plt.show()

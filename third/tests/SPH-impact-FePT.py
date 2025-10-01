import math
import numpy as np
import matplotlib.pyplot as plt

from simpleSPH.data import field, Storage
from simpleSPH import particles
from simpleSPH.eos import HugoniotTwoPhase
from simpleSPH.solver import WendlandC2, RiemannSolver, SPHSolver as Solver
from simpleSPH.stepper import RK1 as Stepper
from time import time
from scipy.optimize import minimize


def main(rho_alpha_eps_start, rho_alpha_eps_end, rho_eps_alpha_start, rho_eps_alpha_end):
    # Параметры задачи impact
    nelem = 1200  # число ячеек
    LImpactor = 6.3  # мм
    LTarget = 6.3  # мм
    materialImpactor = 1  # id материала
    materialTarget = 2  # id материала
    RhoImpactor = 7900  # плотность ударника
    RhoTarget = 7900  # плотность мишени
    С0Impactor = 4570  # скорость звука ударник
    С0Target = 4570  # скорость звука мишень
    GrImpactor = 2.0  # параметр Грюнайзена ударника
    GrTarget = 2.0  # параметр Грюнайзена мишени
    sImpactor = 1.49  # параметр ударной адиабаты ударника
    sTarget = 1.49  # параметр ударной адиабаты мишени
    vImpactor = 1150  # скорость ударника м/c
    tend = 4e-3  # время окончания моделирования мс

    impactorEos = HugoniotTwoPhase(
        rho01=7900, rho02=8300,
        rho12from=rho_alpha_eps_start, rho12to=rho_alpha_eps_end,
        rho21from=rho_eps_alpha_start, rho21to=rho_eps_alpha_end,
        c01=4570, c02=4570,
        s1=1.49, s2=1.40,
        gamma1=2.0, gamma2=2.0
    )
    targetEos = HugoniotTwoPhase(
        rho01=7900, rho02=8300,
        rho12from=rho_alpha_eps_start, rho12to=rho_alpha_eps_end,
        rho21from=rho_eps_alpha_start, rho21to=rho_eps_alpha_end,
        c01=4570, c02=4570,
        s1=1.49, s2=1.40,
        gamma1=2.0, gamma2=2.0
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
            field.material,
            field.phase
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
    elements[field.phase] = np.ones(nelem)  # alpha phase

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
        particles[field.phase][iImpactor] = impactorEos.phase(particles[field.density][iImpactor],
                                                              particles[field.innerEnergy][iImpactor],
                                                              particles[field.phase][iImpactor])
        particles[field.phase][iTarget] = targetEos.phase(particles[field.density][iTarget],
                                                          particles[field.innerEnergy][iTarget],
                                                          particles[field.phase][iTarget])
        particles[field.pressure][iImpactor] = impactorEos.pressure(particles[field.density][iImpactor],
                                                                    particles[field.innerEnergy][iImpactor],
                                                                    particles[field.phase][iImpactor])
        particles[field.soundSpeed][iImpactor] = impactorEos.soundSpeed(particles[field.density][iImpactor],
                                                                        particles[field.innerEnergy][iImpactor],
                                                                        particles[field.phase][iImpactor])
        particles[field.pressure][iTarget] = targetEos.pressure(particles[field.density][iTarget],
                                                                particles[field.innerEnergy][iTarget],
                                                                particles[field.phase][iTarget])
        particles[field.soundSpeed][iTarget] = targetEos.soundSpeed(particles[field.density][iTarget],
                                                                    particles[field.innerEnergy][iTarget],
                                                                    particles[field.phase][iTarget])

    def saveplot(istep):
        plt.close()
        fig, ax1 = plt.subplots()
        ax1.plot(elements.data[field.coords], elements.data[field.density])
        ax1.set_xlim(-LImpactor, 1.5 * LTarget)
        ax2 = ax1.twinx()

        ax2.plot(elements.data[field.coords], elements.data[field.phase])
        plt.savefig("results/%06d.png" % istep)

    # SPH решатель, выбираем тип сглаживающего ядра и
    # метод расчета значений на контактном разрыве
    solver = Solver(contact=RiemannSolver(), kernel=WendlandC2(), smoothing_scale=0.92)
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
        # while istep < 2500:
        # используем генератор step
        time = next(step)
        istep += 1
        # if istep % 100 == 0:
        #     print(istep, time)
        # saveplot(istep)
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

with open("Experiment_data_2_phases.csv") as f:
    title = f.readline()
    data = f.readlines()
data = list(map(lambda x: x.split(','), data[5:]))
ts_exp = list(map(lambda x: float(x[0]) * 1e-3, data))
vs_exp = list(map(lambda x: float(x[1]) * 1e3, data))

min_loss = 100000000
min_p = 0


def fun(x):
    global iters
    rho_alpha_eps_start = x[0]
    rho_alpha_eps_end = x[1]
    rho_eps_alpha_start = x[2]
    rho_eps_alpha_end = x[3]
    visar = main(rho_alpha_eps_start, rho_alpha_eps_end, rho_eps_alpha_start, rho_eps_alpha_end)
    tvisar = visar[:, 0]
    vvisar = visar[:, 1]
    loss = loss_func(ts_exp, vs_exp, tvisar, vvisar)
    iters += 1
    with open("Res_2phases.csv", 'a') as f:
        f.write(f"{x[0]},{x[1]},{x[2]},{x[3]},{loss}\n")
    print("Smth")
    return loss


iters = 0
rho_alpha_eps_start = 8400
rho_alpha_eps_end = 8700
rho_eps_alpha_start = 8500
rho_eps_alpha_end = 8300

with open("Res_2phases.csv", 'w') as f:
    f.write("rho_alpha_eps_start,rho_alpha_eps_end,rho_eps_alpha_start,rho_eps_alpha_end,loss\n")
t1 = time()
# p_opt = minimize(fun, [rho_alpha_eps_start, rho_alpha_eps_end, rho_eps_alpha_start, rho_eps_alpha_end],
#                  bounds=[(8400, 8500), (8650, 8800), (8500, 8650), (8250, 8400)])
# rho_alpha_eps_start, rho_alpha_eps_end, rho_eps_alpha_start, rho_eps_alpha_end = p_opt.x
# print(p_opt)

rho_alpha_eps_start, rho_alpha_eps_end, rho_eps_alpha_start, rho_eps_alpha_end = 8400.0, 8700.0, 8500.0, 8300.0

t2 = time()
print(f"Optimizing time: {t2 - t1}")

t1 = time()
visar = main(rho_alpha_eps_start, rho_alpha_eps_end, rho_eps_alpha_start, rho_eps_alpha_end)
tvisar = visar[:, 0]
vvisar = visar[:, 1]
loss = loss_func(ts_exp, vs_exp, tvisar, vvisar)
t2 = time()
print(f"One iter time: {t2 - t1}")

print(
    f"Found optimal rho_alpha_eps_start = {rho_alpha_eps_start}, rho_alpha_eps_end = {rho_alpha_eps_end}, rho_eps_alpha_start = {rho_eps_alpha_start}, rho_eps_alpha_end = {rho_eps_alpha_end} with loss {loss}")
print(f"Iters = {iters}")
print("True parameters: rho12from = 8440, rho12to = 8717, rho21from = 8550, rho21to = 8312")
# строим график для плотности
# plt.plot(elements.data[field.coords], elements.data[field.velocity])
# plt.xlim(-LImpactor, 1.5*LTarget)

plt.plot(tvisar, vvisar)
plt.scatter(ts_exp, vs_exp)
plt.xlabel("t, s* 1e-3")
plt.ylabel("v, m/s")
plt.show()

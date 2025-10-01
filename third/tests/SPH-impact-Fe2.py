import math
import numpy as np
from time import time

from simpleSPH.data import field, Storage
from simpleSPH import particles
from simpleSPH.eos import Hugoniot, Hugoniot2
from simpleSPH.solver import WendlandC2, RiemannSolver, SPHSolver as Solver
from simpleSPH.stepper import RK1 as Stepper
from scipy.optimize import minimize


def main(rho_alpha_eps_start, rho_alpha_eps_end, rho_eps_alpha_start, rho_eps_alpha_end):
    # Параметры задачи impact
    nelem = 1200  # число ячеек
    LImpactor = 6.3  # мм
    LTarget = 6.3  # мм
    materialImpactor = 1  # id материала
    materialTarget = 2  # id материала
    RhoImpactor_alpha = 7900  # плотность ударника
    RhoImpactor_eps = 8300  # плотность ударника
    RhoTarget_alpha = 7900  # плотность мишени
    RhoTarget_eps = 8300  # плотность мишени
    С0Impactor_alpha = 4570  # скорость звука ударник
    С0Impactor_eps = 4570  # скорость звука ударник
    С0Target_alpha = 4570  # скорость звука мишень
    С0Target_eps = 4570  # скорость звука мишень
    GrImpactor = 2  # параметр Грюнайзена ударника
    GrTarget = 2  # параметр Грюнайзена мишени
    sImpactor_alpha = 1.49  # параметр ударной адиабаты ударника
    sImpactor_eps = 1.4  # параметр ударной адиабаты ударника
    sTarget_alpha = 1.49  # параметр ударной адиабаты мишени
    sTarget_eps = 1.4  # параметр ударной адиабаты мишени
    vImpactor = 671  # скорость ударника м/c
    tend = 4e-3  # время окончания моделирования мс

    # начальное распределение плотности
    def density0(x):
        rho = np.zeros_like(x)
        rho[x < 0] = RhoImpactor_alpha
        rho[x >= 0] = RhoTarget_alpha
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
    iImpactor = np.where(elements[field.material] == materialImpactor)
    iTarget = np.where(elements[field.material] == materialTarget)

    phase_alpha = np.array(['a' for _ in range(len(elements[field.density]))])
    phase_alpha_impactor = phase_alpha[iImpactor]
    phase_alpha_target = phase_alpha[iTarget]

    # impactorEos = Hugoniot2(
    #     gamma=GrImpactor,
    #     rho10=RhoImpactor_alpha,
    #     c10=С0Impactor_alpha,
    #     s1=sImpactor_alpha,
    #     rho20=RhoImpactor_eps,
    #     c20=С0Impactor_eps,
    #     s2=sImpactor_eps,
    #     curr_phase=phase_alpha_impactor,
    #     rhoae_start=rho_alpha_eps_start,
    #     rhoae_end=rho_alpha_eps_end,
    #     rhoea_start=rho_eps_alpha_start,
    #     rhoea_end=rho_eps_alpha_end
    # )
    #
    # targetEos = Hugoniot2(
    #     gamma=GrImpactor,
    #     rho10=RhoImpactor_alpha,
    #     c10=С0Impactor_alpha,
    #     s1=sImpactor_alpha,
    #     rho20=RhoImpactor_eps,
    #     c20=С0Impactor_eps,
    #     s2=sImpactor_eps,
    #     curr_phase=phase_alpha_target,
    #     rhoae_start=rho_alpha_eps_start,
    #     rhoae_end=rho_alpha_eps_end,
    #     rhoea_start=rho_eps_alpha_start,
    #     rhoea_end=rho_eps_alpha_end
    # )
    impactorEos = Hugoniot(
        rho0=RhoImpactor_alpha,
        c0=С0Impactor_alpha,
        s=sImpactor_alpha,
        gamma=GrImpactor
    )
    targetEos = Hugoniot(
        rho0=RhoImpactor_alpha,
        c0=С0Impactor_alpha,
        s=sImpactor_alpha,
        gamma=GrTarget
    )

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
    with open("Res_2phases_my.csv", 'a') as f:
        f.write(f"{x[0]},{x[1]},{x[2]},{x[3]},{loss}\n")
    print("Smth")
    return loss


iters = 0
rho_alpha_eps_start = 8400
rho_alpha_eps_end = 8700
rho_eps_alpha_start = 8500
rho_eps_alpha_end = 8300
with open("Res_2phases_my.csv", 'w') as f:
    f.write("rho_alpha_eps_start,rho_alpha_eps_end,rho_eps_alpha_start,rho_eps_alpha_end,loss\n")
t1 = time()
p_opt = minimize(fun, [rho_alpha_eps_start, rho_alpha_eps_end, rho_eps_alpha_start, rho_eps_alpha_end])
rho_alpha_eps_start, rho_alpha_eps_end, rho_eps_alpha_start, rho_eps_alpha_end = p_opt.x

t2 = time()
print(p_opt)
print(f"Optimizing time: {t2 - t1}")

t1 = time()
visar = main(rho_alpha_eps_start, rho_alpha_eps_end, rho_eps_alpha_start, rho_eps_alpha_end)
tvisar = visar[:, 0]
vvisar = visar[:, 1]
loss = loss_func(ts_exp, vs_exp, tvisar, vvisar)
t2 = time()
print(f"One iter time: {t2 - t1}")

print(f"Found optimal rho_alpha_eps_start = {rho_alpha_eps_start}, rho_alpha_eps_end = {rho_alpha_eps_end}, rho_eps_alpha_start = {rho_eps_alpha_start}, rho_eps_alpha_end = {rho_eps_alpha_end} with loss {loss}")
print(f"Iters = {iters}")
# строим график для плотности
# plt.plot(elements.data[field.coords], elements.data[field.velocity])
# plt.xlim(-LImpactor, 1.5*LTarget)

plt.plot(tvisar, vvisar)
plt.scatter(ts_exp, vs_exp)
plt.xlabel("t, s* 1e-3")
plt.ylabel("v, m/s")
plt.show()

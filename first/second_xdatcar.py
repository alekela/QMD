import matplotlib.pyplot as plt
import numpy as np


with open("XDATCAR.txt") as f:
    data = f.readlines()
title = data[:7]
cell_size = [title[2].strip().split()[0], title[3].strip().split()[1], title[4].strip().split()[2]]
cell_size = list(map(lambda x: float(x) * 1e-10, cell_size))
timestep = 2e-15
n_atoms = int(title[6])
mu = 55.8 / 1000
R = 8.31

n_timesteps = int((len(data) - 7) / (n_atoms + 1))
T = []
for i in range(1, n_timesteps):
    velocities = [0 for _ in range(n_atoms)]
    rads = []
    main_point = data[7 + i * (n_atoms + 1) + 1].strip().split()
    for k in range(1, n_atoms):
        tmp_rad = 0
        point = data[7 + i * (n_atoms + 1) + 1 + k].strip().split()
        for index in range(3):
            tmp_rad += (float(main_point[index]) - float(point[index])) ** 2
        rads.append(tmp_rad ** 0.5)
    dr = max(rads) / 100
    rs = [dr * i for i in range(101)]
    ns = [0 for _ in range(101)]
    for q in rads:
        ns[int(q / dr)] += 1
    ns = np.array(ns) / n_atoms
    if i == 1:
        plt.plot(np.array(rs) * cell_size[0], ns)
        plt.show()
        
    for k in range(n_atoms):
        first_index = 7 + (i - 1) * (n_atoms + 1) + 1 + k
        second_index = 7 + i * (n_atoms + 1) + 1 + k
        first_point = list(map(float, data[first_index].strip().split()))
        second_point = list(map(float, data[second_index].strip().split()))
        for index in range(3):
            if first_point[index] < 0.2 and second_point[index] > 0.8:
                first_point[index] += 1
            elif first_point[index] > 0.8 and second_point[index] < 0.2:
                second_point[index] += 1
            velocities[k] += ((second_point[index] - first_point[index]) * cell_size[index]) ** 2
    v_sr = sum(velocities) / len(velocities)
    v_sr /= timestep ** 2
    T.append(v_sr * mu / 3 / R)

time = [i * timestep for i in range(1, n_timesteps)]
plt.plot(time, T)
plt.show()


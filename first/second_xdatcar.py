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
rho = 7800
gr_discrete = 100
r_max = 8 * 1e-10

n_timesteps = int((len(data) - 7) / (n_atoms + 1))
T = []
for i in range(1, n_timesteps):
    velocities = [0 for _ in range(n_atoms)]
    if i == 1 or i == n_timesteps - 1:
        rads = []
        for k in range(n_atoms):
            point1 = data[7 + i * (n_atoms + 1) + 1 + k].strip().split()
            for q in range(n_atoms):
                if k != q:
                    point2 = data[7 + i * (n_atoms + 1) + 1 + q].strip().split()
                    tmp_rad = 0
                    for index in range(3):
                        tmp_rad += (float(point1[index]) - float(point2[index])) ** 2
                    tmp_rad = tmp_rad ** 0.5 * cell_size[0]
                    if tmp_rad < r_max:
                        rads.append(tmp_rad)

        dr = r_max / (gr_discrete - 1)
        rs = np.array([dr * i for i in range(1, gr_discrete + 1)])
        ns = [0 for _ in range(1, gr_discrete + 1)]
        for q in rads:
            ns[int(q / dr)] += 2
        ns = np.array(ns)
        print(ns)
        # ns = ns / 4 / np.pi / rs ** 2 / dr * cell_size[0] ** 3 / n_atoms / (n_atoms - 1) * 2
        g = ns * cell_size[0] ** 3 / n_atoms ** 2 / (4 * np.pi * rs ** 2 * dr)
        # plt.plot(np.array(rs) * cell_size[0], ns)
        plt.plot(np.array(rs), g)
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


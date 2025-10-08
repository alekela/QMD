import numpy as np

m = 9.1e-31
h = 1.05e-34


def khi(x, n, a):
    return x ** n * (x ** 2 - a ** 2)


def H(x, n, m, a):
    if (n + m) % 2 != 0:
        return 0
    return - 1 / 2 * (
            (m + 2) * (m + 1) / (n + m + 3) * x ** (n + m + 3) - a ** 2 * ((m - 1) * m + (m + 1) * (m + 2)) / (
            n + m + 1) * x ** (n + m + 1) + a ** 4 * m * (m - 1) * x ** (n + m - 1) / (n + m - 1))


def S(x, n, m, a):
    return x ** (n + m + 5) / (n + m + 5) + a ** 4 * x ** (n + m + 1) / (n + m + 1) - 2 * a ** 2 * x ** (n + m + 3) / (
                n + m + 3)


N = 5
a = 1
matrix_H = np.zeros((N, N))
matrix_s = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        matrix_H[i][j] = H(a, i, j, a) - H(-a, i, j, a)
        matrix_s[i][j] = S(a, i, j, a) - S(-a, i, j, a)

# (H - ES)C = 0 - поиск собственных значений

E = np.linalg.eigvals(matrix_H * np.linalg.inv(matrix_s))
print(E)

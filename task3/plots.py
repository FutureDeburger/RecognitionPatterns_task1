import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# -----------------------------
# Общие функции
# -----------------------------

def compute_C_inv(C):
    M = C.shape[2]
    C_inv = np.zeros_like(C)
    for i in range(M):
        C_inv[:, :, i] = np.linalg.inv(C[:, :, i])
    return C_inv


def total_error(P, pw):
    M = len(pw)
    return sum(pw[i] * sum(P[i, j] for j in range(M) if j != i) for i in range(M))


# -----------------------------
# Теоретическая ошибка (PIJ)
# -----------------------------

def theoretical_matrix(m, C, pw):
    n, M = m.shape
    C_inv = compute_C_inv(C)

    PIJ = np.zeros((M, M))

    for i in range(M):
        for j in range(i + 1, M):

            dm = m[:, i] - m[:, j]
            l0 = np.log(pw[j] / pw[i])

            mg = 0.5 * dm.T @ C_inv[:, :, i] @ dm
            Dg = dm.T @ C_inv[:, :, i] @ dm

            PIJ[i, j] = norm.cdf(l0, loc=mg, scale=np.sqrt(Dg))
            PIJ[j, i] = 1 - norm.cdf(l0, loc=-mg, scale=np.sqrt(Dg))

    for i in range(M):
        PIJ[i, i] = 1 - np.sum(PIJ[i, :])

    return PIJ


# -----------------------------
# Чернов (Bhattacharyya)
# -----------------------------

def chernov_error(m, C, pw):
    M = len(pw)
    mu_b = np.zeros((M, M))

    for i in range(M):
        for j in range(i + 1, M):
            dm = m[:, i] - m[:, j]

            C_avg = (C[:, :, i] + C[:, :, j]) / 2
            dti = np.linalg.det(C[:, :, i])
            dtj = np.linalg.det(C[:, :, j])

            mu = (1/8) * dm.T @ np.linalg.inv(C_avg) @ dm \
                 + 0.5 * np.log(np.linalg.det(C_avg) / np.sqrt(dti * dtj))

            mu_b[i, j] = mu
            mu_b[j, i] = mu

    Pe = 0
    for i in range(M):
        for j in range(i + 1, M):
            Pe += np.sqrt(pw[i] * pw[j]) * np.exp(-mu_b[i, j])

    return Pe


# -----------------------------
# Эксперимент
# -----------------------------

def experimental_error(m, C, pw, K):
    M = len(pw)
    C_inv = compute_C_inv(C)

    Pc = np.zeros((M, M))

    for _ in range(K):
        i = np.random.choice(M, p=pw)
        x = np.random.multivariate_normal(m[:, i], C[:, :, i])

        u = []
        for j in range(M):
            val = -0.5 * (x - m[:, j]).T @ C_inv[:, :, j] @ (x - m[:, j]) \
                  - 0.5 * np.log(np.linalg.det(C[:, :, j])) \
                  + np.log(pw[j])
            u.append(val)

        j_hat = np.argmax(u)
        Pc[i, j_hat] += 1

    Pc /= K
    return total_error(Pc, pw)


# -----------------------------
# Исходные данные
# -----------------------------

n = 2
M = 3

pw = np.array([0.6, 0.2, 0.2])
pw = pw / pw.sum()

m_base = np.array([[4, -2], [3, 2], [4, 1]]).T

C = np.zeros((n, n, M))
C[:, :, 0] = np.array([[2, -0.2], [-0.2, 1]])
C[:, :, 1] = np.array([[3, -1.0], [-1.0, 3]])
C[:, :, 2] = np.array([[3, 1.5], [1.5, 1]])

# -----------------------------
# 📈 ГРАФИК 1: ошибка vs K
# -----------------------------

Ks = np.arange(200, 5000, 400)

exp_err = []
theor_err = []
chernov_err = []

for K in Ks:
    exp_err.append(experimental_error(m_base, C, pw, K))

    PIJ = theoretical_matrix(m_base, C, pw)
    theor_err.append(total_error(PIJ, pw))

    chernov_err.append(chernov_error(m_base, C, pw))

plt.figure()
plt.plot(Ks, exp_err, label="Эксперимент")
plt.plot(Ks, theor_err, label="Теоретическая")
plt.plot(Ks, chernov_err, label="Чернов")
plt.xlabel("Число испытаний K")
plt.ylabel("Суммарная ошибка")
plt.title("Ошибка vs число испытаний")
plt.legend()
plt.grid()
plt.show()


# -----------------------------
# ГРАФИК 2: ошибка vs расстояние
# -----------------------------

d_vals = np.linspace(0.5, 5, 10)

exp_err2 = []
theor_err2 = []
chernov_err2 = []

for d in d_vals:
    m = m_base.copy()
    m[:, 1] = m[:, 1] + np.array([d, d])  # сдвиг класса

    exp_err2.append(experimental_error(m, C, pw, 3000))

    PIJ = theoretical_matrix(m, C, pw)
    theor_err2.append(total_error(PIJ, pw))

    chernov_err2.append(chernov_error(m, C, pw))

plt.figure()
plt.plot(d_vals, exp_err2, label="Эксперимент")
plt.plot(d_vals, theor_err2, label="Теоретическая")
plt.plot(d_vals, chernov_err2, label="Чернов")
plt.xlabel("Расстояние между классами")
plt.ylabel("Суммарная ошибка")
plt.title("Ошибка vs расстояние между классами")
plt.legend()
plt.grid()
plt.show()
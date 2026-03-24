import numpy as np
from scipy.stats import multivariate_normal, norm
import matplotlib.pyplot as plt


# -----------------------------
# 1. Исходные данные
# -----------------------------

n = 2
M = 3
K = 5000

pw = np.array([0.6, 0.2, 0.2])
pw = pw / pw.sum()

m = np.array([[4, -2], [3, 2], [4, 1]]).T

C = np.zeros((n, n, M))
C[:, :, 0] = np.array([[2, -0.2], [-0.2, 1]])
C[:, :, 1] = np.array([[3, -1.0], [-1.0, 3]])
C[:, :, 2] = np.array([[3, 1.5], [1.5, 1]])

C_inv = np.zeros_like(C)
for i in range(M):
    C_inv[:, :, i] = np.linalg.inv(C[:, :, i])


# -----------------------------
# 2. Теоретические ошибки (приближение)
# -----------------------------

PIJ = np.zeros((M, M))
PIJB = np.zeros((M, M))
mu_b = np.zeros((M, M))  # Bhattacharyya расстояния

for i in range(M):
    for j in range(i + 1, M):

        dm = m[:, i] - m[:, j]

        dti = np.linalg.det(C[:, :, i])
        dtj = np.linalg.det(C[:, :, j])

        # ---- Bhattacharyya distance (ПРАВИЛЬНЫЙ) ----
        C_avg = (C[:, :, i] + C[:, :, j]) / 2

        mu = (1/8) * dm.T @ np.linalg.inv(C_avg) @ dm \
             + 0.5 * np.log(np.linalg.det(C_avg) / np.sqrt(dti * dtj))

        mu_b[i, j] = mu
        mu_b[j, i] = mu

        # ---- Граница Чернова ----
        PIJB[i, j] = np.sqrt(pw[j] / pw[i]) * np.exp(-mu)
        PIJB[j, i] = np.sqrt(pw[i] / pw[j]) * np.exp(-mu)

        # ---- Теоретическая ошибка (гауссовское приближение) ----
        l0 = np.log(pw[j] / pw[i])

        # среднее и дисперсия (упрощённо)
        mg = 0.5 * (dm.T @ C_inv[:, :, i] @ dm)
        Dg = dm.T @ C_inv[:, :, i] @ dm

        PIJ[i, j] = norm.cdf(l0, loc=mg, scale=np.sqrt(Dg))
        PIJ[j, i] = 1 - norm.cdf(l0, loc=-mg, scale=np.sqrt(Dg))

# диагонали
for i in range(M):
    PIJ[i, i] = 1 - np.sum(PIJ[i, :])
    PIJB[i, i] = 1 - np.sum(PIJB[i, :])


# -----------------------------
# 3. Эксперимент
# -----------------------------

Pc = np.zeros((M, M))

for k in range(K):
    # выбираем класс по pw
    i = np.random.choice(M, p=pw)

    x = np.random.multivariate_normal(m[:, i], C[:, :, i])

    u = np.zeros(M)

    for j in range(M):
        u[j] = -0.5 * (x - m[:, j]).T @ C_inv[:, :, j] @ (x - m[:, j]) \
               - 0.5 * np.log(np.linalg.det(C[:, :, j])) \
               + np.log(pw[j])

    j_hat = np.argmax(u)
    Pc[i, j_hat] += 1

Pc /= K


# -----------------------------
# 4. Суммарные ошибки
# -----------------------------

def total_error(P):
    return sum(pw[i] * sum(P[i, j] for j in range(M) if j != i) for i in range(M))

Pe_exp = total_error(Pc)
Pe_theor = total_error(PIJ)

# Чернов (сумма по парам)
Pe_chernov = 0
for i in range(M):
    for j in range(i + 1, M):
        Pe_chernov += np.sqrt(pw[i] * pw[j]) * np.exp(-mu_b[i, j])


# -----------------------------
# 5. Вывод
# -----------------------------

np.set_printoptions(precision=4, suppress=True)

print("\nЭкспериментальная матрица (Pc):")
print(Pc)

print("\nТеоретическая матрица (PIJ):")
print(PIJ)

print("\nМатрица Чернова (PIJB):")
print(PIJB)

print("\nСуммарные ошибки:")
print(f"Экспериментальная: {Pe_exp:.4f}")
print(f"Теоретическая:     {Pe_theor:.4f}")
print(f"Чернова (оценка):  {Pe_chernov:.4f}")


# -----------------------------
# 6. Визуализация
# -----------------------------

if n == 2:

    x1 = np.linspace(-2, 8, 200)
    x2 = np.linspace(-6, 6, 200)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.column_stack([X1.ravel(), X2.ravel()])

    plt.figure(figsize=(8, 6))

    # плотности
    for i in range(M):
        rv = multivariate_normal(mean=m[:, i], cov=C[:, :, i])
        Z = rv.pdf(X).reshape(X1.shape)
        plt.contour(X1, X2, Z, levels=5)

    # классификация области
    Z_class = np.zeros(X.shape[0])

    for idx, x in enumerate(X):
        u = []
        for j in range(M):
            val = -0.5 * (x - m[:, j]).T @ C_inv[:, :, j] @ (x - m[:, j]) \
                  - 0.5 * np.log(np.linalg.det(C[:, :, j])) \
                  + np.log(pw[j])
            u.append(val)
        Z_class[idx] = np.argmax(u)

    Z_class = Z_class.reshape(X1.shape)

    plt.contourf(X1, X2, Z_class, alpha=0.2)

    plt.title("Классы и границы решений")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.gca().set_aspect('equal')

    plt.show()
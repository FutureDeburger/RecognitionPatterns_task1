import numpy as np
from scipy.stats import multivariate_normal, norm
import matplotlib.pyplot as plt



# -----------------------------
# 1. Исходные данные
# -----------------------------
n = 2 # размерность признакового пространства
M = 3 # число классов
K = 1000 # количество статистических испытаний

# априорные вероятности классов
pw = np.array([0.6, 0.2, 0.2])
#pw = np.array([0.6, 0.4])
pw = pw / pw.sum()


# Математические ожидания классов
m = np.array([[-3, 5], [-5, -5], [5, -3]]).T


#m = np.array([[1, 2, 3],
# [0, 0, 0]]).T


# Матрицы ковариации для классов:
C = np.zeros((n, n, M))
C_ = np.zeros_like(C)

C[:, :, 0] = np.array([[6, -2], [-2, 7]])
C[:, :, 1] = np.array([[6, 2], [2, 6]])
C[:, :, 2] = np.array([[6, -2], [-2, 7]])


#C[:, :, 0] = np.array([[5, 2, 2],

# [2, 2, 0],

# [2, 0, 6]])

#C[:, :, 1] = np.array([[5, -2, -2],

# [-2, 2, 0],

# [-2, 0, 6]])


# Обратные ковариационные матрицы

for k in range(M):
    C_[:, :, k] = np.linalg.inv(C[:, :, k])



# -----------------------------
# 2. Теоретические матрицы ошибок
# -----------------------------

PIJ = np.zeros((M, M)) # теоретическая матрица ошибок
PIJB = np.zeros((M, M)) # матрица границ Чернова
l0_ = np.zeros((M, M))


for i in range(M):
    for j in range(i + 1, M):
        dmij = m[:, i] - m[:, j]
        l0_[i, j] = np.log(pw[j] / pw[i])

        dti = np.linalg.det(C[:, :, i])
        dtj = np.linalg.det(C[:, :, j])

        # Расчёт следов (trace) с разными ковариациями
        trij = np.trace(C_[:, :, j] @ C[:, :, i] - np.eye(n))
        trji = np.trace(np.eye(n) - C_[:, :, i] @ C[:, :, j])
        trij_2 = np.trace(np.linalg.matrix_power((C_[:, :, j] @ C[:, :, i] - np.eye(n)), 2))
        trji_2 = np.trace(np.linalg.matrix_power((np.eye(n) - C_[:, :, i] @ C[:, :, j]), 2))


        # Средние значения и дисперсии решающих функций

        mg1 = 0.5 * (trij + dmij.T @ C_[:, :, i] @ dmij - np.log(dti / dtj))
        Dg1 = 0.5 * trij_2 + dmij.T @ C_[:, :, j] @ C[:, :, i] @ C_[:, :, j] @ dmij

        mg2 = 0.5 * (trji - dmij.T @ C_[:, :, j] @ dmij + np.log(dtj / dti))
        Dg2 = 0.5 * trji_2 + dmij.T @ C_[:, :, i] @ C[:, :, j] @ C_[:, :, i] @ dmij

        sD1 = np.sqrt(Dg1)
        sD2 = np.sqrt(Dg2)


        PIJ[i, j] = norm.cdf(l0_[i, j], loc=mg1, scale=sD1)
        PIJ[j, i] = 1 - norm.cdf(l0_[i, j], loc=mg2, scale=sD2)


        # Расчёт границы Чернова (Bhattacharyya)
        mu2 = (1 / 8) * dmij.T @ np.linalg.inv((C[:, :, i] + C[:, :, j]) / 2) @ dmij \
        + 0.5 * np.log((dti + dtj) / (2 * np.sqrt(dti * dtj)))

        PIJB[i, j] = np.sqrt(pw[j] / pw[i]) * np.exp(-mu2)
        PIJB[j, i] = np.sqrt(pw[i] / pw[j]) * np.exp(-mu2)


    # Нижняя граница вероятности правильного распознавания

    PIJ[i, i] = 1 - np.sum(PIJ[i, :])
    PIJB[i, i] = 1 - np.sum(PIJB[i, :])



# -----------------------------
# 3. Тестирование методом статистических испытаний
# -----------------------------

Pc_ = np.zeros((M, M)) # экспериментальная матрица ошибок

for k in range(K):
    for i in range(M):
        x = np.random.multivariate_normal(m[:, i], C[:, :, i])
        u = np.zeros(M)
        for j in range(M):
            # Решающие функции учитывают разные ковариации:
            u[j] = -0.5 * (x - m[:, j]).T @ C_[:, :, j] @ (x - m[:, j]) \
            - 0.5 * np.log(np.linalg.det(C[:, :, j])) + np.log(pw[j])

        iai = np.argmax(u)
        Pc_[i, iai] += 1

Pc_ /= K


# -----------------------------
# 4. Вывод результатов
# -----------------------------

np.set_printoptions(precision=4, suppress=True)
print("\nЭкспериментальная матрица вероятностей ошибок (Pc_):")
print(Pc_)
print("Теоретическая матрица вероятностей ошибок (PIJ):")
print(PIJ)
print("\nМатрица ошибок на основе границы Чернова (PIJB):")
print(PIJB)



# -----------------------------
# 5. Визуализация (2D)
# -----------------------------

if n == 2:

    Es1 = pw[0] * PIJ[0, 1] + pw[1] * PIJ[1, 0]
    Es2 = np.sqrt(pw[0] * pw[1]) * np.exp(-mu2)
    Es3 = pw[0] * Pc_[0, 1] + pw[1] * Pc_[1, 0]
    print("\nОценки суммарных ошибок:")
    print(f"Теоретическая = {Es1:.4f}, Чернова = {Es2:.4f}, Экспериментальная = {Es3:.4f}")


    D = 3 * np.eye(2)
    xmin1 = -3 * np.sqrt(np.max(D[0, :])) + np.min(m[0, :])
    xmax1 = 3 * np.sqrt(np.max(D[0, :])) + np.max(m[0, :])
    xmin2 = -3 * np.sqrt(np.max(D[1, :])) + np.min(m[1, :])
    xmax2 = 3 * np.sqrt(np.max(D[1, :])) + np.max(m[1, :])
    x1 = np.arange(xmin1, xmax1, 0.1)
    x2 = np.arange(xmin2, xmax2, 0.1)
    X1, X2 = np.meshgrid(x1, x2)
    x12 = np.column_stack([X1.ravel(), X2.ravel()])

    plt.figure(figsize=(8, 6))

    for i in range(M):
        rv = multivariate_normal(mean=m[:, i], cov=C[:, :, i])
        f2 = rv.pdf(x12)
        f3 = f2.reshape(len(x2), len(x1))
        cs = plt.contour(x1, x2, f3, levels=[0.01, 0.5 * f3.max()], colors=['b'], linewidths=0.75)
        plt.clabel(cs, inline=True, fontsize=8)

        for j in range(i + 1, M):
            wij = C_[:, :, i] @ m[:, i] - C_[:, :, j] @ m[:, j]
            wij0 = -0.5 * (m[:, i].T @ C_[:, :, i] @ m[:, i] - m[:, j].T @ C_[:, :, j] @ m[:, j])
            f4 = x12 @ wij + wij0 - 0.5 * np.log(np.linalg.det(C[:, :, i]) / np.linalg.det(C[:, :, j]))
            fd = -0.5 * (C_[:, :, i] - C_[:, :, j]) @ x12.T
            f4 = f4 + np.sum(x12.T * fd, axis=0)
            f5 = f4.reshape(len(x2), len(x1))
            plt.contour(x1, x2, f5, levels=[l0_[i, j]], colors='k', linewidths=1.25)


    plt.title("Области локализации классов и разделяющие границы")
    plt.xlabel("x1"); plt.ylabel("x2")
    strv1 = ' pw='; strv2 = ' '.join([f'{val: .2g}' for val in pw])
    plt.text(xmin1 + 1, xmax2 - 1, strv1 + strv2, horizontalalignment='left', backgroundcolor=[.8, .8, .8], fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.legend(['wi', 'gij(x)=0'])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

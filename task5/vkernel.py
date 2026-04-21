#####
##### Для метода Парзена
#####


import numpy as np

def vkernel(x, XN, h_N, kl_kernel):
    # Функция для отображения ядра и получения оценки
    # плотности распределения вероятностей методом Парзена
    # x-массив векторов (точек), для которых определяется плотность
    # XN-входная обучающая выборка данных
    # h_N- параметр, определяющий размер области локализации ядра (оконной функции)
    # kl_kernel-ключ для определения вида используемого ядра
    # kl_kernel=11 - гауссовская функция c использованием диагональной матрицы;
    # kl_kernel=12 - гауссовская функция c использованием матрицы ковариации;
    # kl_kernel=2 - показательная функция;
    # kl_kernel=3 - оконная прямоугольная функция;
    # kl_kernel=4 - оконная треугольная функция;
    n1, mx = x.shape  # размерность пространства и число точек, для которых проводится оценка
    n2, N = XN.shape  # размерность пространства и объем обучающей выборки
    if n1 == n2:
        n = n1
        p_ = np.zeros(mx)
        C = np.eye(n)
        C_ = C
        if kl_kernel == 12 and N > 1:
            # оценка выборочной матрицы ковариаций
            C = np.zeros((n, n))
            m_ = np.mean(XN, axis=1).reshape(-1, 1)
            for i in range(N):
                diff = (XN[:, i].reshape(-1, 1) - m_)
                C += diff @ diff.T
            C /= (N - 1)
            C_ = np.linalg.inv(C)

        # Вычисление значений функций ядра с центрами XN[:, 0:N] в точках x[:, 0:mx]
        fit = np.zeros((N, mx))
        for i in range(N):
            p_k = np.zeros((n, mx))
            mx_i = np.tile(XN[:, i].reshape(-1, 1), (1, mx))
            if kl_kernel == 11:
                ro = np.sum((x - mx_i) ** 2, axis=0)
                fit[i, :] = np.exp(-ro / (2 * h_N ** 2)) / ((2 * np.pi) ** (n / 2) * (h_N ** n))
            elif kl_kernel == 12:
                ro = np.sum((C_ @ (x - mx_i)) * (x - mx_i), axis=0)
                fit[i, :] = np.exp(-ro / (2 * h_N ** 2)) * ((2 * np.pi) ** (-n / 2) * (h_N ** -n) * (np.linalg.det(C) ** -0.5))
            elif kl_kernel == 2:
                ro = np.abs(x - mx_i) / h_N
                fit[i, :] = np.prod(np.exp(-ro), axis=0) / (2 * h_N ** n)
            elif kl_kernel == 3:
                ro = np.abs(x - mx_i) / h_N
                for k in range(n):
                    ind = ro[k, :] < 1
                    p_k[k, ind] = 1 / 2
                fit[i, :] = np.prod(p_k, axis=0) / h_N ** n
            elif kl_kernel == 4:
                ro = np.abs(x - mx_i) / h_N
                for k in range(n):
                    ind = ro[k, :] < 1
                    p_k[k, ind] = (1 - ro[k, ind])
                fit[i, :] = np.prod(p_k, axis=0) / h_N ** n
        # Вычисление оценки плотности распределения вероятностей
        if N > 1:
            p_ = np.sum(fit, axis=0) / N
        else:
            p_ = fit[0, :]
    else:
        raise ValueError('размерности данных (n1 и n2) не совпадают')
    return p_

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats



# m1 = [-5, 2, -3]
# m2 = [-4, -5, 4]
# C = [[11, -0.8, 1.0], [-0.8, 10, 0.5], [1.0, 0.5, 7]]


#1. Задание исходных данных
n, M = 3, 2 #размерность признакового пространства и число классов
K = 1000 #количество статистических испытаний

# мат. ожидания - координаты центров классов
m = np.array([[-5, 2, -3], [-4, -5, 4]])

# априорные вероятности классов (доля образов каждого класса в общей выборке)
pw = np.array([0.5, 0.5])
pw = pw / sum(pw) # нормировка априорных вероятностей

#матрица ковариаций классов
C = np.array([[11, -0.8, 1.0], [-0.8, 10, 0.5], [1.0, 0.5, 7]])
C_ = np.linalg.inv(C) #обратная матрица ковариаций


# 1.1. Визуализация исходной совокупности образов
# Определение числа образов в каждом классе, пропорцианально pw fix(pw .* K)
Ks = np.round(pw * K).astype(int)
Ks[-1] = K - sum(Ks[:-1]) #чтобы сумма была равна K
# print(Ks)
# label = ['bo', 'r+', 'k*', 'gx'] #маркеры классов для визуализации
label = ['bo', 'r+']
IMS = np.empty((n, 0)) #общая совокупность образов (общая выборка)
plt.figure()
# Задание осей на случай 3D
if n == 3:
    ax = plt.axes(projection='3d')
plt.title('Исходные метки образов')
for i in range(M): #цикл по классам
    #генерация Ks(i) образов i-го класса
    ims = np.random.multivariate_normal(m[i], C, Ks[i]).T
    if n == 2:
        plt.plot(ims[0, :], ims[1, :], label[i], markersize=8, linewidth=1)
    elif n == 3:
        ax.plot3D(ims[0, :], ims[1, :], ims[2, :], label[i], markersize=8, linewidth=1)
    IMS = np.hstack((IMS, ims)) #добавление в общую совокупность образов
plt.grid()
plt.xlabel('X1')
plt.ylabel('X2')
if n == 3:
    ax.set_zlabel('X3')
# plt.legend()
# plt.axis('equal')



# 2. Расчет разделяющих функций и матрицы вероятностей ошибок распознавания
G = np.zeros((M, n + 1)); PIJ = np.zeros((M, M)); l0_ = np.zeros((M, M))
for i in range(M):
    G[i, 0:n] = (C_ @ m[i]).T
    G[i, n] = -0.5 * m[i].T @ C_ @ m[i]
    for j in range(i + 1, M):
        l0_[i, j] = np.log(pw[j] / pw[i])
        h = 0.5 * (m[i] - m[j]).T @ C_ @ (m[i] - m[j])
        sD = np.sqrt(2 * h)
        PIJ[i, j] = stats.norm.cdf(l0_[i, j], h, sD)
        PIJ[j, i] = 1 - stats.norm.cdf(l0_[i, j], -h, sD)
    # PIJ - теоретическая матрица ошибок
    # нижняя граница вероятности правильного распознавания
    # (на главной диагонали)
    PIJ[i, i] = 1 - sum(PIJ[i, :])

# 2.1. Визуализация результатов распознавания образов
plt.figure()
# Задание осей на случай 3D
if n == 3:
    ax = plt.axes(projection='3d')
plt.title('Результат классификации образов')
for i in range(K): #цикл по всем образам совокупности
    z = np.hstack((IMS[:, i], 1)) #значение очередного образа из общей совокупности
    u = G @ z + np.log(pw) #вычисление значения разделяющих функций
    iai = np.argmax(u) #определение максимума (iai - индекс класса)
    if n == 2:
        plt.plot(IMS[0, i], IMS[1, i], label[iai], markersize=8, linewidth=1)
    elif n == 3:
        ax.plot3D(IMS[0, i], IMS[1, i], IMS[2, i], label[iai], markersize=8, linewidth=1)
plt.grid()
plt.xlabel('X1')
plt.ylabel('X2')
if n == 3:
    ax.set_zlabel('X3')
# plt.legend()
# plt.axis('equal')



#3. Тестирование алгоритма методом статистических испытаний
x = np.ones((n + 1, 1)); Pc_ = np.zeros((M, M))  # экспериментальная матрица вероятностей ошибок
for k in range(K):  # цикл по числу испытаний
    for i in range(M):  # цикл по классам
        # генерация образа i-го класса
        x[0:n, 0] = np.random.multivariate_normal(m[i], C)
        u = G @ x + np.log(pw).reshape(M, 1)  # вычисление значения разделяющих функций
        iai = np.argmax(u)  # определение максимума
        Pc_[i, iai] = Pc_[i, iai] + 1  # фиксация результата распознавания
Pc_ = Pc_ / K  # матрица ошибок, полученная экспериментально
# у нее такая же структура как и в PIJ, только вычисляется численно, а не по формулам
print('Теоретическая матрица ошибок PIJ')
print(np.round(PIJ, 3))
print('Экспериментальная матрица ошибок Pc_')
print(np.round(Pc_, 3))



#4. Визуализация областей принятия решений для двумерного случая
# if n == 2:
#     D = 1
#     xmin1 = -4 * np.sqrt(D) + min(m[:, 0]); xmax1 = 4 * np.sqrt(D) + max(m[:, 0])
#     xmin2 = -4 * np.sqrt(D) + min(m[:, 1]); xmax2 = 4 * np.sqrt(D) + max(m[:, 1])
#     x1 = np.arange(xmin1, xmax1, 0.05); x2 = np.arange(xmin2, xmax2, 0.05)
#     plt.figure(); plt.grid(); plt.title('Области локализации классов и разделяющие границы')
#     # установка границ поля графика по осям
#     plt.axis([xmin1, xmax1, xmin2, xmax2])
#     # матрицы значений координат случайного вектора
#     X1, X2 = np.meshgrid(x1, x2)
#     x12 = np.column_stack((X1.ravel(), X2.ravel()))
#     for i in range(M):
#         # массив значений плотности распределения
#         f2 = stats.multivariate_normal.pdf(x12, mean=m[i], cov=C)
#         # матрица значений плотности распределения
#         f3 = f2.reshape(len(x2), len(x1))
#         CS = plt.contour(x1, x2, f3, levels=[0.01, 0.5 * np.max(f3)], colors='b', linewidths=0.75)
#         plt.clabel(CS, inline=1, fontsize=10)
#         for j in range(i + 1, M):  # изображение разделяющих границ
#             wij = C_ @ (m[i] - m[j])
#             wij0 = -0.5 * (m[i] + m[j]).T @ C_ @ (m[i] - m[j])
#             f4 = x12 @ wij + wij0
#             f5 = f4.reshape(len(x2), len(x1))
#             CS_ = plt.contour(x1, x2, f5, levels=[l0_[i, j] + 0.0001], colors='k', linewidths=1.25)
#     plt.xlabel('x1'); plt.ylabel('x2')
#     strv1 = ' pw='; strv2 = ' '.join([f'{val: .2g}' for val in pw])
#     plt.text(xmin1 + 1, xmax2 - 1, strv1 + strv2, horizontalalignment='left', backgroundcolor=[.8, .8, .8], fontsize=12)
#     plt.legend(['wi', 'gij(x)=0'])
#     plt.gca().set_aspect('equal', adjustable='box')



plt.show()
import numpy as np

import matplotlib.pyplot as plt

import scipy.stats as stats

from scipy.stats import uniform

# 1. Генерация СВ

# число реализаций Гауссовской случайной величины (СВ)

K = 2000

# число реализаций равномерной СВ для генерации одной реализации гауссовской СВ

n = 12

# параметры распределения

mu = -9

sig = 8

# теоретическое среднее и дисперсия из таблицы

m = mu

d = sig ** 2

# генерация alpha (равномерная СВ) и реализаций Гауссовской СВ

alf = uniform.rvs(size=(K, n))

x = sig * (np.sum(alf, axis=1) - 6) + mu

# Вычисление выборочных дисперсии

# (матожидание реализовать самим по аналогии)

ds = np.zeros(K)

# Мат ожидание:
me = np.zeros(K)

for k in range(K):
    ds[k] = np.var(x[:k + 1], ddof=1)
    me[k] = np.mean(x[:k + 1])

# print(f'Выборочная дисперсия при K={K}: {ds[-1]}')

# print(f'Теоретическая дисперсия: {d}')


# создание графического окна

plt.figure()
# отображение зависимости выборочных дисперсий от числа реализаций СВ

plt.plot(range(1, K + 1), ds, color='blue', label='Выборочная дисперсия')

plt.axhline(y=d, color='red', linestyle='--', label='Теоретическая дисперсия')
plt.title('Выборочная дисперсия от числа реализаций СВ')
plt.xlabel('Число реализаций СВ')
plt.ylabel('Выборочная дисперсия')

plt.legend()
plt.grid()

# Отображение мат ожидания от числа реализаций СВ

plt.figure()
# отображение зависимости выборочных дисперсий от числа реализаций СВ

plt.plot(range(1, K + 1), me, color='blue', label='Выборочная дисперсия')

plt.axhline(y=m, color='red', linestyle='--', label='Теоретическое матожидание')
plt.title('мат ожидание от числа реализаций СВ')
plt.xlabel('Число реализаций СВ')
plt.ylabel('Выборочное мат.ожидание')

plt.legend()
plt.grid()

# 3. Гистограмма и плотность распределения

# создание второго графического окна
plt.figure()

bins = 20
# отображение гистограммы (исправлено: x вместо x_norm, добавлена нормализация density=True)
plt.hist(x, bins=bins, density=True, color='lightblue', edgecolor='black', label='Гистограмма выборки')
# создание массива значений для плотности распределения
x_vals = np.linspace(mu - 4 * sig, mu + 4 * sig, 1000)

# вычисление плотности распределения
pdf = stats.norm.pdf(x_vals, mu, sig)
# отображение плотности распределения
plt.plot(x_vals, pdf, color='red', linewidth=2, label='Плотность нормального распределения')
plt.title('Гистограмма выборки и плотность нормального распределения')
plt.xlabel('Значение СВ')
plt.ylabel('Плотность')
plt.legend()
plt.grid()


delta = np.zeros(K)

for k in range(10, K + 1):  # начинаем хотя бы с 10 значений
    hist, bin_edges = np.histogram(x[:k], bins=bins, density=True)

    # центры бинов
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # теоретическая плотность в центрах
    pdf_theor = stats.norm.pdf(bin_centers, mu, sig)

    # средняя абсолютная разность
    delta[k - 1] = np.mean(np.abs(hist - pdf_theor))

# построение графика
plt.figure()
plt.plot(range(1, K + 1), delta)
plt.title('Средняя абсолютная разность\nэмпирической и теоретической плотности')
plt.xlabel('Число реализаций СВ')
plt.ylabel('Средняя абсолютная разность')
plt.grid()


# один раз в конце для отображения всех графиков
plt.show()

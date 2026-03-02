import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


# 1. Генерация СВ

# число реализаций
K = 2000

# параметры распределения
v = 3
w = 3

# теоретическое среднее и дисперсия из таблицы
m = v / (v + w)
d = (v * w) / ((v + w)**2 * (v + w + 1))

# генерация выборки
x = stats.beta.rvs(v, w, size=K)


# 2. Выборочное матожидание

ms = np.zeros(K)

for k in range(K):
    ms[k] = np.mean(x[:k + 1])

plt.figure()
plt.plot(range(1, K + 1), ms, label='Выборочное мат. ожидание')
plt.axhline(y=m, linestyle='--', label='Теоретическое мат. ожидание')

plt.title('Матожидание от числа реализаций')
plt.xlabel('Количество реализаций')
plt.ylabel('Матожидание')
plt.legend()
plt.grid()


# 3. Выборочная дисперсия

ds = np.zeros(K)

for k in range(K):
    ds[k] = np.var(x[:k + 1], ddof=1)

plt.figure()
plt.plot(range(1, K + 1), ds, label='Выборочная дисперсия')
plt.axhline(y=d, linestyle='--', label='Теоретическая дисперсия')

plt.title('Дисперсия от числа реализаций')
plt.xlabel('Количество реализаций')
plt.ylabel('Дисперсия')
plt.legend()
plt.grid()


# 4. Гистограмма + плотность

plt.figure()
sns.histplot(x, bins=25, stat='density', color='lightblue', label='Гистограмма')

# теоретическая плотность
t = np.linspace(0, 1, 1000)
pdf = stats.beta.pdf(t, v, w)

plt.plot(t, pdf, color='red', label='Теоретическая плотность')
plt.title('Гистограмма и плотность бета-распределения')
plt.xlabel('x')
plt.ylabel('Плотность')
plt.legend()
plt.grid()


# 5. Средняя абсолютная разность плотностей

diff = np.zeros(K)
bins = np.linspace(0, 1, 40)

for k in range(50, K):
    hist, edges = np.histogram(x[:k], bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    theoretical = stats.beta.pdf(centers, v, w)

    diff[k] = np.mean(np.abs(hist - theoretical))

plt.figure()
plt.plot(range(1, K + 1), diff)

plt.title('Средняя абсолютная разность плотностей')
plt.xlabel('Количество реализаций')
plt.ylabel('Ошибка')
plt.grid()
plt.show()
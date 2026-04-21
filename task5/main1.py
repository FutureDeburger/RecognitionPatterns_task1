import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.neighbors import NearestNeighbors
import math


# ============== Копируем твои функции vkernel и vknn ==============

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
                fit[i, :] = np.exp(-ro / (2 * h_N ** 2)) * (
                            (2 * np.pi) ** (-n / 2) * (h_N ** -n) * (np.linalg.det(C) ** -0.5))
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


def vknn(x, XN, k):
    # Функция для получения оценки плотности распределения вероятностей
    # методом k-ближайших соседей
    # x-массив векторов (точек), для которых проводится оценка плотности
    # XN-входная обучающая выборка данных
    # k - число ближайших соседей
    n1, mx = x.shape  # размерность пространства и число точек, для которых проводится оценка
    n2, N = XN.shape  # размерность пространства и объем обучающей выборки
    if n1 == n2 and k <= N:
        n = n1
        p_ = np.zeros(mx)
        Cn = 2 * (np.pi ** (n / 2)) / (n * math.gamma(n / 2))
        # Вычисление mx значений плотности для каждой точки x[:, 0:mx]
        nbrs = NearestNeighbors(n_neighbors=k).fit(XN.T)
        distances, indices = nbrs.kneighbors(x.T)
        r = distances[:, -1]  # расстояния до k-го соседа
        # расчет объема гипершаров радиусов r[:, 0] в пространстве размерности n
        V = Cn * (r ** n)
        p_ = (k / N) / V
    else:
        if n1 != n2:
            raise ValueError('размерности данных (n1 и n2) не совпадают')
        if k > N:
            raise ValueError('число соседей больше количества обучающих векторов')
    return p_


# ============== Основная часть программы ==============

# Параметры эксперимента
K = 1000  # размер обучающей выборки (для ускорения вычислений)
v = 3  # параметр бета-распределения
w = 3  # параметр бета-распределения

# Генерация выборки
np.random.seed(42)  # для воспроизводимости
x_sample = stats.beta.rvs(v, w, size=K)

# Точки, в которых будем оценивать плотность
n_points = 200
x_eval = np.linspace(0.01, 0.99, n_points).reshape(1, -1)

# Теоретическая плотность
true_pdf = stats.beta.pdf(x_eval.flatten(), v, w)

# Преобразование выборки для функций vkernel и vknn
XN = x_sample.reshape(1, -1)

# ============== Задание 1: Метод Парзена ==============
print("Вычисление ошибок для метода Парзена...")

# Параметры для исследования
h_values = np.logspace(-2, 0, 50)  # значения параметра h
kernels = {
    11: 'Гауссовская (диаг. матрица)',
    2: 'Показательная',
    3: 'Прямоугольная'
}

errors_parzen = {kernel: [] for kernel in kernels}

for kl_kernel, kernel_name in kernels.items():
    print(f"  Обработка ядра: {kernel_name}")
    for h in h_values:
        try:
            # Оценка плотности методом Парзена
            estimated_pdf = vkernel(x_eval, XN, h, kl_kernel)

            # Вычисление средней абсолютной ошибки
            mae = np.mean(np.abs(estimated_pdf - true_pdf))
            errors_parzen[kl_kernel].append(mae)
        except Exception as e:
            print(f"    Ошибка при h={h:.4f}: {e}")
            errors_parzen[kl_kernel].append(np.nan)

# ============== Задание 2: Метод k ближайших соседей ==============
print("\nВычисление ошибок для метода kNN...")

# Значения k (степени двойки)
k_values = [2, 4, 8, 16, 32, 64, 128, 256, 512]
errors_knn = []

for k in k_values:
    if k <= K:
        try:
            # Оценка плотности методом kNN
            estimated_pdf = vknn(x_eval, XN, k)

            # Вычисление средней абсолютной ошибки
            mae = np.mean(np.abs(estimated_pdf - true_pdf))
            errors_knn.append(mae)
            print(f"  k={k}: MAE={mae:.6f}")
        except Exception as e:
            print(f"  Ошибка при k={k}: {e}")
            errors_knn.append(np.nan)
    else:
        errors_knn.append(np.nan)

# ============== Построение графиков ==============

# График 1: Зависимость ошибки от параметра h для разных оконных функций
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
colors = ['blue', 'green', 'red']
markers = ['o', 's', '^']
for (kl_kernel, kernel_name), color, marker in zip(kernels.items(), colors, markers):
    # Убираем NaN значения для построения
    valid_idx = ~np.isnan(errors_parzen[kl_kernel])
    h_valid = h_values[valid_idx]
    err_valid = np.array(errors_parzen[kl_kernel])[valid_idx]
    plt.semilogx(h_valid, err_valid, 'o-', color=color, marker=marker, markersize=3, linewidth=1.5, label=kernel_name)

plt.xlabel('Параметр окна h', fontsize=12)
plt.ylabel('Средняя абсолютная ошибка (MAE)', fontsize=12)
plt.title('Метод Парзена: зависимость ошибки от h', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# График 2: Зависимость ошибки от k для метода kNN
plt.subplot(1, 2, 2)
valid_idx_knn = ~np.isnan(errors_knn)
k_valid = [k_values[i] for i in range(len(k_values)) if valid_idx_knn[i]]
err_valid_knn = [errors_knn[i] for i in range(len(errors_knn)) if valid_idx_knn[i]]

plt.semilogx(k_valid, err_valid_knn, 'o-', color='purple', marker='d', markersize=5, linewidth=1.5, label='Метод k ближайших соседей')
plt.xlabel('Количество соседей k', fontsize=12)
plt.ylabel('Средняя абсолютная ошибка (MAE)', fontsize=12)
plt.title('Метод kNN: зависимость ошибки от k', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('error_plots.png', dpi=150, bbox_inches='tight')
plt.show()

# ============== Поиск лучших параметров ==============

# Находим лучший параметр h для каждого ядра
print("\n" + "=" * 50)
print("Лучшие результаты для метода Парзена:")
best_parzen_results = {}
for kl_kernel, kernel_name in kernels.items():
    err_array = np.array(errors_parzen[kl_kernel])
    valid_idx = ~np.isnan(err_array)
    if np.any(valid_idx):
        best_idx = np.argmin(err_array[valid_idx])
        actual_idx = np.where(valid_idx)[0][best_idx]
        best_h = h_values[actual_idx]
        best_err = err_array[actual_idx]
        best_parzen_results[kl_kernel] = (best_h, best_err)
        print(f"  {kernel_name}: h_opt = {best_h:.4f}, MAE = {best_err:.6f}")

# Находим лучшее k для метода kNN
best_idx_knn = np.argmin([e for e in errors_knn if not np.isnan(e)])
best_k = k_values[best_idx_knn]
best_err_knn = errors_knn[best_idx_knn]
print(f"\nЛучший результат для метода kNN:")
print(f"  k_opt = {best_k}, MAE = {best_err_knn:.6f}")

# ============== Графики плотностей для лучших случаев ==============

plt.figure(figsize=(14, 5))

# Подграфик 1: Метод Парзена - лучший результат
plt.subplot(1, 2, 1)
plt.plot(x_eval.flatten(), true_pdf, 'k-', linewidth=2, label='Теоретическая плотность')

# Находим лучшее ядро среди всех
best_overall_kernel = min(best_parzen_results.items(),
                          key=lambda x: x[1][1])[0]
best_h_overall = best_parzen_results[best_overall_kernel][0]
best_pdf_parzen = vkernel(x_eval, XN, best_h_overall, best_overall_kernel)

plt.plot(x_eval.flatten(), best_pdf_parzen, 'r--', linewidth=2,
         label=f'Оценка Парзена (ядро: {kernels[best_overall_kernel]}, h={best_h_overall:.3f})')

plt.xlabel('x', fontsize=12)
plt.ylabel('Плотность', fontsize=12)
plt.title('Метод Парзена - лучший результат', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# Подграфик 2: Метод kNN - лучший результат
plt.subplot(1, 2, 2)
plt.plot(x_eval.flatten(), true_pdf, 'k-', linewidth=2, label='Теоретическая плотность')

best_pdf_knn = vknn(x_eval, XN, best_k)
plt.plot(x_eval.flatten(), best_pdf_knn, 'b--', linewidth=2,
         label=f'Оценка kNN (k={best_k})')

plt.xlabel('x', fontsize=12)
plt.ylabel('Плотность', fontsize=12)
plt.title('Метод k ближайших соседей - лучший результат', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('best_density_plots.png', dpi=150, bbox_inches='tight')
plt.show()

# ============== Сравнительный график всех лучших методов ==============

plt.figure(figsize=(10, 6))
plt.plot(x_eval.flatten(), true_pdf, 'k-', linewidth=2, label='Теоретическая плотность')

# Все лучшие оценки Парзена
colors_line = ['red', 'green', 'blue']
for (kl_kernel, kernel_name), color in zip(kernels.items(), colors_line):
    best_h = best_parzen_results[kl_kernel][0]
    best_pdf = vkernel(x_eval, XN, best_h, kl_kernel)
    plt.plot(x_eval.flatten(), best_pdf, '--', color=color, linewidth=1.5,
             label=f'Парзен ({kernel_name}, h={best_h:.3f})')

# Лучшая оценка kNN
plt.plot(x_eval.flatten(), best_pdf_knn, '-.', color='purple', linewidth=2,
         label=f'kNN (k={best_k})')

plt.xlabel('x', fontsize=12)
plt.ylabel('Плотность', fontsize=12)
plt.title('Сравнение всех методов оценивания плотности', fontsize=14)
plt.legend(loc='best', fontsize=9)
plt.grid(True, alpha=0.3)
plt.savefig('all_methods_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============== Вывод результатов в консоль ==============
print("\n" + "=" * 50)
print("РЕЗЮМЕ:")
print("=" * 50)
print(f"Распределение: Бета({v}, {w})")
print(f"Объем выборки: {K}")
print("\nМетод Парзена:")
for kl_kernel, kernel_name in kernels.items():
    best_h, best_err = best_parzen_results[kl_kernel]
    print(f"  {kernel_name:30} : h_opt={best_h:.4f}, MAE={best_err:.6f}")
print(f"\nМетод k ближайших соседей:")
print(f"  Лучшее k={best_k}, MAE={best_err_knn:.6f}")

print("\nНаилучший метод по MAE:")
all_errors = [(f"Парзен ({kernels[best_overall_kernel]})", best_parzen_results[best_overall_kernel][1]),
              (f"kNN", best_err_knn)]
best_method = min(all_errors, key=lambda x: x[1])
print(f"  {best_method[0]} с ошибкой {best_method[1]:.6f}")
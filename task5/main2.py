import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings('ignore')


# ============== Функция Парзена ==============

def vkernel(x, XN, h_N, kl_kernel):
    n1, mx = x.shape
    n2, N = XN.shape
    if n1 == n2:
        n = n1
        p_ = np.zeros(mx)
        C = np.eye(n)
        C_ = C
        if kl_kernel == 12 and N > 1:
            C = np.cov(XN)
            # Добавляем небольшое возмущение для устойчивости
            C = C + np.eye(n) * 1e-6
            C_ = np.linalg.inv(C)

        fit = np.zeros((N, mx))
        for i in range(N):
            p_k = np.zeros((n, mx))
            mx_i = np.tile(XN[:, i].reshape(-1, 1), (1, mx))
            if kl_kernel == 11:  # Гауссовское с диагональной матрицей
                ro = np.sum((x - mx_i) ** 2, axis=0)
                fit[i, :] = np.exp(-ro / (2 * h_N ** 2)) / ((2 * np.pi) ** (n / 2) * (h_N ** n))
            elif kl_kernel == 12:  # Гауссовское с матрицей ковариаций
                ro = np.sum((C_ @ (x - mx_i)) * (x - mx_i), axis=0)
                fit[i, :] = np.exp(-ro / (2 * h_N ** 2)) * (
                            (2 * np.pi) ** (-n / 2) * (h_N ** -n) * (np.linalg.det(C) ** -0.5))
            elif kl_kernel == 2:  # Показательное
                ro = np.abs(x - mx_i) / h_N
                fit[i, :] = np.prod(np.exp(-ro), axis=0) / (2 * h_N ** n)
            elif kl_kernel == 3:  # Прямоугольное
                ro = np.abs(x - mx_i) / h_N
                for k in range(n):
                    ind = ro[k, :] < 1
                    p_k[k, ind] = 1 / 2
                fit[i, :] = np.prod(p_k, axis=0) / h_N ** n
            elif kl_kernel == 4:  # Треугольное
                ro = np.abs(x - mx_i) / h_N
                for k in range(n):
                    ind = ro[k, :] < 1
                    p_k[k, ind] = (1 - ro[k, ind])
                fit[i, :] = np.prod(p_k, axis=0) / h_N ** n
        if N > 1:
            p_ = np.sum(fit, axis=0) / N
        else:
            p_ = fit[0, :]
    else:
        raise ValueError('размерности данных не совпадают')
    return p_


# ============== Функция k ближайших соседей ==============

def vknn(x, XN, k):
    n1, mx = x.shape
    n2, N = XN.shape
    if n1 == n2 and k <= N:
        n = n1
        p_ = np.zeros(mx)
        Cn = 2 * (np.pi ** (n / 2)) / (n * math.gamma(n / 2))
        nbrs = NearestNeighbors(n_neighbors=k).fit(XN.T)
        distances, indices = nbrs.kneighbors(x.T)
        r = distances[:, -1]
        # Избегаем деления на ноль
        r[r < 1e-10] = 1e-10
        V = Cn * (r ** n)
        p_ = (k / N) / V
        p_[V == 0] = 0
    else:
        if n1 != n2:
            raise ValueError('размерности данных не совпадают')
        if k > N:
            raise ValueError('число соседей больше количества обучающих векторов')
    return p_


# ============== 1. Формирование данных из лаб 2 и 3 ==============

print("=" * 70)
print("ЗАДАНИЕ 2: Оценивание плотности распределения по средним значениям из лаб 2 и 3")
print("=" * 70)

# Данные из лабораторной работы №2
m_lab2 = np.array([[-5, 2, -3], [-4, -5, 4]])

# Отбрасываем третье измерение
m_lab2_2d = m_lab2[:, :2]
print("\n1. Лабораторная работа №2 (после отбрасывания 3-го измерения):")
print(f"   Класс 1: m1 = [{m_lab2_2d[0, 0]:.1f}, {m_lab2_2d[0, 1]:.1f}]")
print(f"   Класс 2: m2 = [{m_lab2_2d[1, 0]:.1f}, {m_lab2_2d[1, 1]:.1f}]")

# Данные из лабораторной работы №3
m_lab3 = np.array([[4, -2], [3, 2], [4, 1]])
print("\n2. Лабораторная работа №3:")
print(f"   Класс 1: m1 = [{m_lab3[0, 0]:.1f}, {m_lab3[0, 1]:.1f}]")
print(f"   Класс 2: m2 = [{m_lab3[1, 0]:.1f}, {m_lab3[1, 1]:.1f}]")
print(f"   Класс 3: m3 = [{m_lab3[2, 0]:.1f}, {m_lab3[2, 1]:.1f}]")

# Объединяем все средние значения (всего 5 точек в 2D)
m_all = np.vstack([m_lab2_2d, m_lab3])
print(f"\n3. Общая выборка средних значений (всего {len(m_all)} точек):")
for i, point in enumerate(m_all):
    print(f"   Точка {i + 1}: [{point[0]:.1f}, {point[1]:.1f}]")

XN = m_all.T

# ============== 2. Создание сетки для оценки плотности ==============

x_min = np.min(m_all[:, 0]) - 4
x_max = np.max(m_all[:, 0]) + 4
y_min = np.min(m_all[:, 1]) - 4
y_max = np.max(m_all[:, 1]) + 4

print(f"\nОбласть оценивания: x ∈ [{x_min:.1f}, {x_max:.1f}], y ∈ [{y_min:.1f}, {y_max:.1f}]")

grid_size = 60
x_grid = np.linspace(x_min, x_max, grid_size)
y_grid = np.linspace(y_min, y_max, grid_size)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
x_eval = np.vstack([X_grid.ravel(), Y_grid.ravel()])

# ============== 3. Метод Парзена ==============

print("\n" + "=" * 70)
print("МЕТОД ПАРЗЕНА")
print("=" * 70)

# Более широкий диапазон h для малого количества точек
h_values = np.logspace(-0.5, 2, 40)

kernels = {
    11: 'Гауссовская (диаг. матрица)',
    12: 'Гауссовская (ковариац. матрица)',
    2: 'Показательная',
    3: 'Прямоугольная'
}

errors_parzen = {k: [] for k in kernels}


# Создаем опорную плотность - сумму гауссиан вокруг каждой точки
def reference_density(x, XN, sigma=1.0):
    """Опорная плотность как сумма гауссиан вокруг каждой точки"""
    n, N = XN.shape
    p = np.zeros(x.shape[1])
    for i in range(N):
        diff = x - XN[:, i].reshape(-1, 1)
        ro = np.sum(diff ** 2, axis=0)
        p += np.exp(-ro / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    return p / N


# Вычисляем опорную плотность с небольшим sigma
p_reference = reference_density(x_eval, XN, sigma=0.5)

for kl_kernel, kernel_name in kernels.items():
    print(f"\nИсследуется ядро: {kernel_name}")

    for h in h_values:
        try:
            p_est = vkernel(x_eval, XN, h, kl_kernel)
            # Сравниваем с опорной плотностью
            mae = np.mean(np.abs(p_est - p_reference))
            errors_parzen[kl_kernel].append(mae)
        except Exception as e:
            errors_parzen[kl_kernel].append(np.nan)

    valid_errors = [e for e in errors_parzen[kl_kernel] if not np.isnan(e)]
    if valid_errors:
        min_err = np.min(valid_errors)
        print(f"  Минимальная ошибка: {min_err:.6f}")

# ============== 4. Метод k ближайших соседей ==============

print("\n" + "=" * 70)
print("МЕТОД k БЛИЖАЙШИХ СОСЕДЕЙ")
print("=" * 70)

k_values = [1, 2, 4]
k_values = [k for k in k_values if k <= XN.shape[1]]
print(f"Значения k: {k_values}")

errors_knn = []

for k in k_values:
    try:
        p_est = vknn(x_eval, XN, k)
        mae = np.mean(np.abs(p_est - p_reference))
        errors_knn.append(mae)
        print(f"  k={k}: MAE={mae:.6f}")
    except Exception as e:
        print(f"  Ошибка при k={k}: {e}")
        errors_knn.append(np.nan)

# ============== 5. Построение графиков ==============

# График 1: Зависимость ошибки от h для метода Парзена
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
colors = ['blue', 'darkblue', 'green', 'red']
linestyles = ['-', '--', '-.', ':']
markers = ['o', 's', '^', 'D']

for idx, (kl_kernel, kernel_name) in enumerate(kernels.items()):
    valid_idx = ~np.isnan(errors_parzen[kl_kernel])
    if np.any(valid_idx):
        h_valid = h_values[valid_idx]
        err_valid = np.array(errors_parzen[kl_kernel])[valid_idx]
        plt.semilogx(h_valid, err_valid,
                     linestyle=linestyles[idx],
                     color=colors[idx],
                     marker=markers[idx],
                     markersize=3,
                     linewidth=1.5,
                     label=kernel_name)

plt.xlabel('Параметр окна h', fontsize=12)
plt.ylabel('Средняя абсолютная ошибка (MAE)', fontsize=12)
plt.title('Метод Парзена: зависимость ошибки от h\n(5 точек, 2D пространство)', fontsize=11)
plt.legend(loc='best', fontsize=8)
plt.grid(True, alpha=0.3)

# График 2: Зависимость ошибки от k для метода kNN
plt.subplot(1, 2, 2)
valid_idx_knn = ~np.isnan(errors_knn)
k_valid = [k_values[i] for i in range(len(k_values)) if valid_idx_knn[i]]
err_valid_knn = [errors_knn[i] for i in range(len(errors_knn)) if valid_idx_knn[i]]

if len(k_valid) > 0:
    plt.semilogx(k_valid, err_valid_knn, '-o', color='purple', marker='d',
                 markersize=6, linewidth=2, label='Метод k ближайших соседей')
    plt.xlabel('Количество соседей k', fontsize=12)
    plt.ylabel('Средняя абсолютная ошибка (MAE)', fontsize=12)
    plt.title('Метод kNN: зависимость ошибки от k\n(5 точек, 2D пространство)', fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task2_error_plots.png', dpi=150, bbox_inches='tight')
plt.show()

# ============== 6. Визуализация лучших оценок плотности ==============

print("\n" + "=" * 70)
print("ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ")
print("=" * 70)

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
        print(f"  {kernel_name:35} : h_opt={best_h:.4f}, MAE={best_err:.6f}")

# Визуализация
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, (kl_kernel, kernel_name) in enumerate(kernels.items()):
    if idx >= 5:
        break
    best_h = best_parzen_results[kl_kernel][0]
    p_est = vkernel(x_eval, XN, best_h, kl_kernel)
    P_grid = p_est.reshape(grid_size, grid_size)

    ax = axes[idx]
    contour = ax.contourf(X_grid, Y_grid, P_grid, levels=25, cmap='viridis', alpha=0.8)
    ax.scatter(XN[0, :], XN[1, :], c='red', s=120, edgecolors='black',
               linewidths=2, zorder=5, label='Исходные точки')

    for i, (x, y) in enumerate(XN.T):
        ax.annotate(f'{i + 1}', (x, y), xytext=(5, 5),
                    textcoords='offset points', fontsize=10, fontweight='bold')

    ax.set_title(f'{kernel_name}\nh = {best_h:.3f}', fontsize=10)
    ax.set_xlabel('x₁', fontsize=9)
    ax.set_ylabel('x₂', fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.colorbar(contour, ax=ax, label='Плотность')

# Метод kNN
if len(k_valid) > 0:
    best_idx_knn = np.argmin([e for e in errors_knn if not np.isnan(e)])
    best_k = k_values[best_idx_knn]
    p_est_knn = vknn(x_eval, XN, best_k)
    P_grid_knn = p_est_knn.reshape(grid_size, grid_size)

    ax = axes[5]
    contour = ax.contourf(X_grid, Y_grid, P_grid_knn, levels=25, cmap='plasma', alpha=0.8)
    ax.scatter(XN[0, :], XN[1, :], c='red', s=120, edgecolors='black',
               linewidths=2, zorder=5, label='Исходные точки')

    for i, (x, y) in enumerate(XN.T):
        ax.annotate(f'{i + 1}', (x, y), xytext=(5, 5),
                    textcoords='offset points', fontsize=10, fontweight='bold')

    ax.set_title(f'Метод k ближайших соседей\nk = {best_k}', fontsize=10)
    ax.set_xlabel('x₁', fontsize=9)
    ax.set_ylabel('x₂', fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.colorbar(contour, ax=ax, label='Плотность')

plt.suptitle('Оценки плотности распределения для наилучших параметров', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('task2_density_plots.png', dpi=150, bbox_inches='tight')
plt.show()

# ============== 7. Итоговый вывод ==============
print("\n" + "=" * 70)
print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
print("=" * 70)

best_parzen_method = min(best_parzen_results.items(), key=lambda x: x[1][1])
print(f"\nЛучший метод Парзена: {kernels[best_parzen_method[0]]}")
print(f"  Оптимальный параметр h = {best_parzen_method[1][0]:.4f}")
print(f"  MAE = {best_parzen_method[1][1]:.6f}")

if len(k_valid) > 0:
    best_k_error = min([e for e in errors_knn if not np.isnan(e)])
    print(f"\nЛучший метод kNN:")
    best_k_idx = np.argmin([e for e in errors_knn if not np.isnan(e)])
    print(f"  Оптимальный параметр k = {k_values[best_k_idx]}")
    print(f"  MAE = {best_k_error:.6f}")

print(f"\nСтатистика по исходным точкам (5 точек в 2D):")
print(f"  Центр масс: ({np.mean(XN[0, :]):.2f}, {np.mean(XN[1, :]):.2f})")
print(f"  Разброс по x: [{np.min(XN[0, :]):.1f}, {np.max(XN[0, :]):.1f}]")
print(f"  Разброс по y: [{np.min(XN[1, :]):.1f}, {np.max(XN[1, :]):.1f}]")
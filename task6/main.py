import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm
import math


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def vkernel(x, X, h, kernel_type):
    """Оценка плотности методом Парзена"""
    if X.shape[1] == 0:
        return 0

    n, N = X.shape if len(X.shape) == 2 else (X.shape[0], 1)
    x = x.reshape(-1, 1)

    if kernel_type == 1:  # Гауссовское с диагональной матрицей
        cov_diag = h ** 2 * np.eye(n)
        density = 0
        for i in range(N):
            Xi = X[:, i].reshape(-1, 1)
            diff = x - Xi
            exponent = -0.5 * (diff.T @ np.linalg.inv(cov_diag) @ diff)
            density += np.exp(exponent[0, 0]) / (np.sqrt((2 * np.pi) ** n * np.linalg.det(cov_diag)))
        return density / N

    elif kernel_type == 2:  # Гауссовское с полной ковариацией
        cov_est = np.cov(X) + h ** 2 * np.eye(n)
        density = 0
        rv = multivariate_normal(mean=np.zeros(n), cov=cov_est)
        for i in range(N):
            diff = x.flatten() - X[:, i]
            density += rv.pdf(diff)
        return density / N

    elif kernel_type == 3:  # Показательное
        density = 0
        for i in range(N):
            diff = x - X[:, i].reshape(-1, 1)
            distance = np.sum(np.abs(diff))
            density += np.exp(-distance / h)
        return density / (N * (2 * h) ** n)

    elif kernel_type == 4:  # Прямоугольное
        density = 0
        for i in range(N):
            diff = x - X[:, i].reshape(-1, 1)
            if np.max(np.abs(diff)) <= h / 2:
                density += 1
        return density / (N * h ** n)

    return 0


def vknn(x, X, k):
    """Оценка плотности методом k ближайших соседей"""
    if X.shape[1] == 0 or k > X.shape[1]:
        return 0

    n, N = X.shape if len(X.shape) == 2 else (X.shape[0], 1)
    x = x.reshape(-1, 1)

    distances = []
    for i in range(N):
        diff = x - X[:, i].reshape(-1, 1)
        dist = np.sqrt(np.sum(diff ** 2))
        distances.append(dist)

    distances.sort()
    V_k = (np.pi ** (n / 2) / math.gamma(n / 2 + 1)) * distances[k - 1] ** n

    return k / (N * V_k)


def generate_data(n, M, K, m, C, pw):
    """Генерация обучающих выборок"""
    Ks = np.fix(K * pw).astype(int)
    diff = K - np.sum(Ks)
    if diff != 0:
        Ks[-1] += diff

    XN = []
    for i in range(M):
        XN_i = np.random.multivariate_normal(mean=m[:, i], cov=C[:, :, i], size=Ks[i]).T
        XN.append(XN_i)

    return XN, Ks


def calculate_theoretical_errors(M, m, C, pw):
    """Расчет теоретических вероятностей ошибок"""
    n = m.shape[0]
    C_inv = np.zeros((n, n, M))
    PIJ = np.zeros((M, M))

    for k in range(M):
        C_inv[:, :, k] = np.linalg.inv(C[:, :, k])

    for i in range(M):
        for j in range(i + 1, M):
            dmij = m[:, i] - m[:, j]
            l0 = np.log(pw[j] / pw[i])

            dti = np.linalg.det(C[:, :, i])
            dtj = np.linalg.det(C[:, :, j])

            trij = np.trace(C_inv[:, :, j] @ C[:, :, i] - np.eye(n))
            trji = np.trace(np.eye(n) - C_inv[:, :, i] @ C[:, :, j])

            mg1 = 0.5 * (trij + dmij.T @ C_inv[:, :, j] @ dmij - np.log(dti / dtj))
            Dg1 = 0.5 * trij ** 2 + dmij.T @ C_inv[:, :, j] @ C[:, :, i] @ C_inv[:, :, j] @ dmij
            mg2 = 0.5 * (trji - dmij.T @ C_inv[:, :, i] @ dmij + np.log(dtj / dti))
            Dg2 = 0.5 * trji ** 2 + dmij.T @ C_inv[:, :, i] @ C[:, :, j] @ C_inv[:, :, i] @ dmij

            sD1 = np.sqrt(Dg1) if Dg1 > 0 else 1e-6
            sD2 = np.sqrt(Dg2) if Dg2 > 0 else 1e-6

            PIJ[i, j] = norm.cdf(l0, mg1, sD1)
            PIJ[j, i] = 1 - norm.cdf(l0, mg2, sD2)

        PIJ[i, i] = 1 - np.sum(PIJ[i, :])

    return PIJ


def sliding_control_parzen_fast(XN, Ks, n, M, r, kernel_type, h_values):
    """Быстрый метод скользящего контроля для Парзена"""
    best_h = None
    best_error = np.inf

    for h in h_values:
        Pc1 = np.zeros((M, M))

        for i in range(M):
            N = int(Ks[i])
            if N <= 1:
                continue
            XNi = XN[i]

            for j in range(min(N, 30)):  # Берем только 30 образцов для ускорения
                x = XNi[:, j]
                XNi_train = np.delete(XNi, j, axis=1)

                p_estimates = np.zeros(M)
                p_estimates[i] = vkernel(x, XNi_train, h, kernel_type)

                for t in range(M):
                    if t != i:
                        p_estimates[t] = vkernel(x, XN[t], h, kernel_type)

                class_pred = np.argmax(p_estimates)
                Pc1[i, class_pred] += 1

            if N > 0:
                Pc1[i, :] = Pc1[i, :] / min(N, 30)

        total_error = 1 - np.trace(Pc1) / M
        if total_error < best_error:
            best_error = total_error
            best_h = h

    return best_h, best_error


def sliding_control_knn_fast(XN, Ks, M, k_values):
    """Быстрый метод скользящего контроля для kNN"""
    best_k = None
    best_error = np.inf

    for k in k_values:
        Pc1 = np.zeros((M, M))

        for i in range(M):
            N = int(Ks[i])
            if N <= 1:
                continue
            XNi = XN[i]

            for j in range(min(N, 30)):  # Берем только 30 образцов
                x = XNi[:, j]
                XNi_train = np.delete(XNi, j, axis=1)

                p_estimates = np.zeros(M)
                try:
                    p_estimates[i] = vknn(x, XNi_train, min(k, XNi_train.shape[1]))
                except:
                    p_estimates[i] = 0

                for t in range(M):
                    if t != i:
                        try:
                            p_estimates[t] = vknn(x, XN[t], min(k, XN[t].shape[1]))
                        except:
                            p_estimates[t] = 0

                class_pred = np.argmax(p_estimates) if np.sum(p_estimates) > 0 else 0
                Pc1[i, class_pred] += 1

            if N > 0:
                Pc1[i, :] = Pc1[i, :] / min(N, 30)

        total_error = 1 - np.trace(Pc1) / M
        if total_error < best_error:
            best_error = total_error
            best_k = k

    return best_k, best_error


def experiment_parzen(XN, Ks, M, K_test, m, C, pw, h, kernel_type):
    """Экспериментальное тестирование метода Парзена"""
    n = m.shape[0]
    Pcv = np.zeros((M, M))

    for _ in range(K_test):
        for i in range(M):
            x = np.random.multivariate_normal(mean=m[:, i], cov=C[:, :, i], size=1).T

            p_estimates = np.zeros(M)
            for j in range(M):
                p_estimates[j] = vkernel(x, XN[j], h, kernel_type)

            class_pred = np.argmax(p_estimates)
            Pcv[i, class_pred] += 1

    return Pcv / K_test


def experiment_gaussian(XN, Ks, M, K_test, m, C, pw):
    """Экспериментальное тестирование гауссовского классификатора"""
    n = m.shape[0]
    Pc = np.zeros((M, M))
    C_inv = np.zeros((n, n, M))

    for k in range(M):
        C_inv[:, :, k] = np.linalg.inv(C[:, :, k])

    for _ in range(K_test):
        for i in range(M):
            x = np.random.multivariate_normal(mean=m[:, i], cov=C[:, :, i], size=1).flatten()

            u = np.zeros(M)
            for j in range(M):
                diff = x - m[:, j]
                u[j] = -0.5 * diff.T @ C_inv[:, :, j] @ diff - 0.5 * np.log(np.linalg.det(C[:, :, j])) + np.log(pw[j])

            class_pred = np.argmax(u)
            Pc[i, class_pred] += 1

    return Pc / K_test


def experiment_knn(XN, Ks, M, K_test, m, C, pw, k):
    """Экспериментальное тестирование метода kNN"""
    Pcv = np.zeros((M, M))

    for _ in range(K_test):
        for i in range(M):
            x = np.random.multivariate_normal(mean=m[:, i], cov=C[:, :, i], size=1).T

            p_estimates = np.zeros(M)
            for j in range(M):
                try:
                    p_estimates[j] = vknn(x, XN[j], min(k, XN[j].shape[1]))
                except:
                    p_estimates[j] = 0

            class_pred = np.argmax(p_estimates)
            Pcv[i, class_pred] += 1

    return Pcv / K_test


def plot_error_vs_h(XN, Ks, n, M, r):
    """График зависимости ошибки от параметра h"""
    h_range = np.logspace(-2, 2, 20)
    errors = {1: [], 2: [], 3: [], 4: []}
    kernel_names = {
        1: 'Гауссовское (диаг. матрица)',
        2: 'Гауссовское (полная ковариация)',
        3: 'Показательное',
        4: 'Прямоугольное'
    }

    plt.figure(figsize=(12, 8))

    for kernel_type in [1, 2, 3, 4]:
        print(f"  Вычисление для ядра: {kernel_names[kernel_type]}...")
        for h in h_range:
            _, error = sliding_control_parzen_fast(XN, Ks, n, M, r, kernel_type, [h])
            errors[kernel_type].append(error)

        plt.semilogx(h_range, errors[kernel_type], 'o-', label=kernel_names[kernel_type], linewidth=2, markersize=4)

    plt.xlabel('Параметр окна h', fontsize=12)
    plt.ylabel('Суммарная ошибка распознавания', fontsize=12)
    plt.title('Зависимость ошибки от параметра h для различных ядер (метод Парзена)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('error_vs_h.png', dpi=150)
    plt.show()

    return errors


def plot_error_vs_k(XN, Ks, M):
    """График зависимости ошибки от параметра k"""
    max_k = min(30, int(np.min(Ks) / 2))
    if max_k < 1:
        max_k = 1
    k_range = range(1, max_k + 1)
    errors = []

    plt.figure(figsize=(10, 6))

    print(f"  Вычисление для k от 1 до {max_k}...")
    for k in k_range:
        _, error = sliding_control_knn_fast(XN, Ks, M, [k])
        errors.append(error)

    plt.plot(k_range, errors, 's-', color='red', linewidth=2, markersize=6)
    plt.xlabel('Параметр k (число соседей)', fontsize=12)
    plt.ylabel('Суммарная ошибка распознавания', fontsize=12)
    plt.title('Зависимость ошибки от параметра k (метод k ближайших соседей)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('error_vs_k.png', dpi=150)
    plt.show()

    return k_range, errors


def plot_error_vs_distance(XN, Ks, n, M, K_train, K_test, m_initial, C, pw, best_h, best_k):
    """График зависимости ошибки от расстояния между классами"""
    distances = np.linspace(0.5, 5.0, 8)
    errors_parzen_exp = []
    errors_parzen_cv = []
    errors_knn_exp = []
    errors_knn_cv = []
    errors_gaussian = []

    plt.figure(figsize=(12, 8))

    for scale in distances:
        print(f"  Расстояние (масштаб) = {scale:.2f}...")

        m_scaled = m_initial.copy()
        center = np.mean(m_initial, axis=1, keepdims=True)
        m_centered = m_initial - center
        m_scaled = center + scale * m_centered

        XN_temp, Ks_temp = generate_data(n, M, K_train, m_scaled, C, pw)

        Pc_gaussian = experiment_gaussian(XN_temp, Ks_temp, M, K_test, m_scaled, C, pw)
        errors_gaussian.append(1 - np.mean(np.diag(Pc_gaussian)))

        Pc_parzen = experiment_parzen(XN_temp, Ks_temp, M, K_test, m_scaled, C, pw, best_h, 1)
        errors_parzen_exp.append(1 - np.mean(np.diag(Pc_parzen)))

        Pc_knn = experiment_knn(XN_temp, Ks_temp, M, K_test, m_scaled, C, pw, best_k)
        errors_knn_exp.append(1 - np.mean(np.diag(Pc_knn)))

        _, error_parzen_cv = sliding_control_parzen_fast(XN_temp, Ks_temp, n, M, 0.5, 1, [best_h])
        errors_parzen_cv.append(error_parzen_cv)

        _, error_knn_cv = sliding_control_knn_fast(XN_temp, Ks_temp, M, [best_k])
        errors_knn_cv.append(error_knn_cv)

    # График для Парзена
    plt.subplot(1, 2, 1)
    plt.plot(distances, errors_parzen_exp, 'o-', label='Экспериментальная ошибка', linewidth=2, markersize=8)
    plt.plot(distances, errors_parzen_cv, 's-', label='Скользящий контроль', linewidth=2, markersize=8)
    plt.xlabel('Расстояние между классами (масштаб)', fontsize=11)
    plt.ylabel('Суммарная ошибка', fontsize=11)
    plt.title('Метод Парзена', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # График для kNN
    plt.subplot(1, 2, 2)
    plt.plot(distances, errors_knn_exp, 'o-', label='Экспериментальная ошибка', linewidth=2, markersize=8)
    plt.plot(distances, errors_knn_cv, 's-', label='Скользящий контроль', linewidth=2, markersize=8)
    plt.xlabel('Расстояние между классами (масштаб)', fontsize=11)
    plt.ylabel('Суммарная ошибка', fontsize=11)
    plt.title('Метод k ближайших соседей', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.suptitle('Зависимость ошибки от расстояния между классами', fontsize=14)
    plt.tight_layout()
    plt.savefig('error_vs_distance.png', dpi=150)
    plt.show()

    # Сравнительный график
    plt.figure(figsize=(10, 6))
    plt.plot(distances, errors_gaussian, '^-', label='Гауссовский классификатор', linewidth=2, markersize=8)
    plt.plot(distances, errors_parzen_exp, 'o-', label='Парзен', linewidth=2, markersize=8)
    plt.plot(distances, errors_knn_exp, 's-', label='kNN', linewidth=2, markersize=8)
    plt.xlabel('Расстояние между классами (масштаб)', fontsize=12)
    plt.ylabel('Суммарная ошибка', fontsize=12)
    plt.title('Сравнение методов при различном расстоянии между классами', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('error_vs_distance_comparison.png', dpi=150)
    plt.show()

    return distances, errors_parzen_exp, errors_parzen_cv, errors_knn_exp, errors_knn_cv


def plot_error_vs_variance(XN, Ks, n, M, K_train, K_test, m, C_initial, pw, best_h, best_k):
    """График зависимости ошибки от дисперсии"""
    variance_scales = np.logspace(-0.5, 1, 8)
    errors_parzen_exp = []
    errors_parzen_cv = []
    errors_knn_exp = []
    errors_knn_cv = []
    errors_gaussian = []

    plt.figure(figsize=(12, 8))

    for scale in variance_scales:
        print(f"  Масштаб дисперсии = {scale:.3f}...")

        C_scaled = np.zeros((n, n, M))
        for i in range(M):
            C_scaled[:, :, i] = scale * C_initial[:, :, i]

        XN_temp, Ks_temp = generate_data(n, M, K_train, m, C_scaled, pw)

        Pc_gaussian = experiment_gaussian(XN_temp, Ks_temp, M, K_test, m, C_scaled, pw)
        errors_gaussian.append(1 - np.mean(np.diag(Pc_gaussian)))

        Pc_parzen = experiment_parzen(XN_temp, Ks_temp, M, K_test, m, C_scaled, pw, best_h, 1)
        errors_parzen_exp.append(1 - np.mean(np.diag(Pc_parzen)))

        Pc_knn = experiment_knn(XN_temp, Ks_temp, M, K_test, m, C_scaled, pw, best_k)
        errors_knn_exp.append(1 - np.mean(np.diag(Pc_knn)))

        _, error_parzen_cv = sliding_control_parzen_fast(XN_temp, Ks_temp, n, M, 0.5, 1, [best_h])
        errors_parzen_cv.append(error_parzen_cv)

        _, error_knn_cv = sliding_control_knn_fast(XN_temp, Ks_temp, M, [best_k])
        errors_knn_cv.append(error_knn_cv)

    # График для Парзена
    plt.subplot(1, 2, 1)
    plt.semilogx(variance_scales, errors_parzen_exp, 'o-', label='Экспериментальная ошибка', linewidth=2, markersize=8)
    plt.semilogx(variance_scales, errors_parzen_cv, 's-', label='Скользящий контроль', linewidth=2, markersize=8)
    plt.xlabel('Масштаб дисперсии', fontsize=11)
    plt.ylabel('Суммарная ошибка', fontsize=11)
    plt.title('Метод Парзена', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # График для kNN
    plt.subplot(1, 2, 2)
    plt.semilogx(variance_scales, errors_knn_exp, 'o-', label='Экспериментальная ошибка', linewidth=2, markersize=8)
    plt.semilogx(variance_scales, errors_knn_cv, 's-', label='Скользящий контроль', linewidth=2, markersize=8)
    plt.xlabel('Масштаб дисперсии', fontsize=11)
    plt.ylabel('Суммарная ошибка', fontsize=11)
    plt.title('Метод k ближайших соседей', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.suptitle('Зависимость ошибки от дисперсии признаков', fontsize=14)
    plt.tight_layout()
    plt.savefig('error_vs_variance.png', dpi=150)
    plt.show()

    # Сравнительный график
    plt.figure(figsize=(10, 6))
    plt.semilogx(variance_scales, errors_gaussian, '^-', label='Гауссовский классификатор', linewidth=2, markersize=8)
    plt.semilogx(variance_scales, errors_parzen_exp, 'o-', label='Парзен', linewidth=2, markersize=8)
    plt.semilogx(variance_scales, errors_knn_exp, 's-', label='kNN', linewidth=2, markersize=8)
    plt.xlabel('Масштаб дисперсии', fontsize=12)
    plt.ylabel('Суммарная ошибка', fontsize=12)
    plt.title('Сравнение методов при различной дисперсии', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('error_vs_variance_comparison.png', dpi=150)
    plt.show()

    return variance_scales, errors_parzen_exp, errors_parzen_cv, errors_knn_exp, errors_knn_cv


# ==================== ОСНОВНАЯ ПРОГРАММА ====================

def main():
    print("=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА №6")
    print("=" * 60)

    # Исходные данные
    n, M = 2, 3
    K_train, K_test = 500, 1000

    pw = np.array([0.4, 0.6, 0.5])
    pw = pw / np.sum(pw)

    m = np.array([[4, 1], [-1, -2], [-4, 1]]).T

    C = np.zeros((n, n, M))
    C[:, :, 0] = np.array([[3, -1], [-1, 3]])
    C[:, :, 1] = np.array([[3, 1], [1, 3]])
    C[:, :, 2] = np.array([[3, 2], [2, 3]])

    print("\n--- ИСХОДНЫЕ ДАННЫЕ ---")
    print(f"Априорные вероятности: {pw}")
    print(f"Математические ожидания:\n{m}")

    # Генерация данных
    print("\n--- ГЕНЕРАЦИЯ ОБУЧАЮЩИХ ВЫБОРОК ---")
    XN, Ks = generate_data(n, M, K_train, m, C, pw)
    print(f"Размеры классов: {Ks}")

    # Теоретические ошибки
    print("\n--- ТЕОРЕТИЧЕСКИЕ ОЦЕНКИ ---")
    PIJ = calculate_theoretical_errors(M, m, C, pw)
    print("Теоретическая матрица ошибок:")
    print(np.round(PIJ, 4))

    # Оптимизация параметров
    print("\n--- ОПТИМИЗАЦИЯ ПАРАМЕТРОВ ---")

    h_range = np.logspace(-2, 2, 15)
    best_h, _ = sliding_control_parzen_fast(XN, Ks, n, M, 0.5, 1, h_range)
    print(f"Оптимальное h = {best_h:.4f}")

    k_range = range(1, min(30, int(np.min(Ks) / 2)) + 1)
    best_k, _ = sliding_control_knn_fast(XN, Ks, M, k_range)
    print(f"Оптимальное k = {best_k}")

    # Графики
    print("\n--- ПОСТРОЕНИЕ ГРАФИКОВ ---")
    plot_error_vs_h(XN, Ks, n, M, 0.5)
    plot_error_vs_k(XN, Ks, M)

    # Эксперименты
    print("\n--- ЭКСПЕРИМЕНТАЛЬНОЕ ТЕСТИРОВАНИЕ ---")

    Pc_gauss = experiment_gaussian(XN, Ks, M, K_test, m, C, pw)
    print("\nГауссовский классификатор:")
    print(np.round(Pc_gauss, 4))
    print(f"Ошибка: {1 - np.mean(np.diag(Pc_gauss)):.4f}")

    Pc_parzen = experiment_parzen(XN, Ks, M, K_test, m, C, pw, best_h, 1)
    print("\nМетод Парзена:")
    print(np.round(Pc_parzen, 4))
    print(f"Ошибка: {1 - np.mean(np.diag(Pc_parzen)):.4f}")

    Pc_knn = experiment_knn(XN, Ks, M, K_test, m, C, pw, best_k)
    print("\nМетод kNN:")
    print(np.round(Pc_knn, 4))
    print(f"Ошибка: {1 - np.mean(np.diag(Pc_knn)):.4f}")

    # Графики зависимостей (быстрые)
    print("\n--- ЗАВИСИМОСТЬ ОТ РАССТОЯНИЯ ---")
    plot_error_vs_distance(XN, Ks, n, M, K_train, K_test, m, C, pw, best_h, best_k)

    print("\n--- ЗАВИСИМОСТЬ ОТ ДИСПЕРСИИ ---")
    plot_error_vs_variance(XN, Ks, n, M, K_train, K_test, m, C, pw, best_h, best_k)

    print("\n" + "=" * 60)
    print("ГОТОВО!")
    print("=" * 60)


if __name__ == "__main__":
    main()
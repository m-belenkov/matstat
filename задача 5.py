import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

theta_true = 5
n = 100
alpha = 0.05
np.random.seed(228)

# Генерация выборки
sample = np.random.uniform(theta_true, 2 * theta_true, n)

# Метод моментов
theta_MM = (2 / 3) * np.mean(sample)

# Метод максимального правдоподобия:
M = np.max(sample)
theta_MLE = M / 2

# Исправленная
theta_MLE_corr = ( (n + 1) * M ) / (2 * n + 1)

# Вывод оценок
print("Оценка theta методом моментов: {:.4f}".format(theta_MM))
print("МП-оценка theta (исходная): {:.4f}".format(theta_MLE))
print("Исправленная МП-оценка theta: {:.4f}".format(theta_MLE_corr))

#  Точный доверительный интервал для theta
lower_exact = M / (1 + (1 - alpha) ** (1 / n))
upper_exact = M

print("Точный доверительный интервал для theta: [{:.4f}, {:.4f}]".format(lower_exact, upper_exact))

# Асимптотический доверительный интервал
se = theta_MM / np.sqrt(27 * n)
z = norm.ppf(1 - alpha / 2)
lower_asymp = theta_MM - z * se
upper_asymp = theta_MM + z * se

print("Асимптотический доверительный интервал для theta: [{:.4f}, {:.4f}]".format(lower_asymp, upper_asymp))

# Бутстраповский доверительный интервал
def theta_estimator(sample_arr):
    M_sample = np.max(sample_arr)
    return ((n + 1) * M_sample) / (2 * n + 1)

B = 1000
theta_boot = np.empty(B)

for b in range(B):
    boot_sample = np.random.choice(sample, size=n, replace=True)
    theta_boot[b] = theta_estimator(boot_sample)

# Используем процентильный метод для доверительного интервала
lower_boot = np.percentile(theta_boot, 100 * (alpha / 2))
upper_boot = np.percentile(theta_boot, 100 * (1 - alpha / 2))

print("Бутстраповский доверительный интервал для theta: [{:.4f}, {:.4f}]".format(lower_boot, upper_boot))

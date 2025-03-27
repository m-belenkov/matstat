import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import stats, special

np.random.seed(0)
n = 25
sample = np.random.exponential(scale=1, size=n)

# (a) Вычисления: мода, медиана, размах, коэффициент асимметрии
rounded_sample = np.round(sample, 2)
mode_result = stats.mode(rounded_sample)
mode_val = mode_result.mode
if isinstance(mode_val, np.ndarray):
    mode_val = mode_val[0]

median_val = np.median(sample)
range_val = np.max(sample) - np.min(sample)
skew_val = stats.skew(sample, bias=False)

print("Пункт (a):")
print("Мода:", mode_val)
print("Медиана:", median_val)
print("Размах:", range_val)
print("Коэффициент асимметрии:", skew_val)

# (b) Графики: эмпирическая функция распределения, гистограмма, boxplot
fig, ax = plt.subplots(1, 3, figsize=(15, 4))

# Эмпирическая функция распределения
sorted_sample = np.sort(sample)
ecdf = np.arange(1, n+1) / n
ax[0].step(sorted_sample, ecdf, where="post")
ax[0].set_title("Эмпирическая ФР")
ax[0].set_xlabel("x")
ax[0].set_ylabel("F(x)")

# Гистограмма
ax[1].hist(sample, bins='auto', edgecolor='black')
ax[1].set_title("Гистограмма")
ax[1].set_xlabel("x")
ax[1].set_ylabel("Частота")

# Boxplot
ax[2].boxplot(sample, vert=False)
ax[2].set_title("Boxplot")
ax[2].set_xlabel("x")

plt.tight_layout()
plt.show()

# (c) Сравнение плотности распределения среднего: ЦПТ vs бутстрап
B = 1000
boot_means = np.array([np.mean(np.random.choice(sample, size=n, replace=True)) for _ in range(B)])
m_sample = np.mean(sample)
s_sample = np.std(sample, ddof=1)
se = s_sample / np.sqrt(n)

x_vals = np.linspace(min(boot_means) - 0.2, max(boot_means) + 0.2, 200)
clt_density = stats.norm.pdf(x_vals, loc=m_sample, scale=se)
kde = stats.gaussian_kde(boot_means)
boot_density = kde(x_vals)

plt.figure(figsize=(6,4))
plt.plot(x_vals, clt_density, label="ЦПТ (норма)")
plt.plot(x_vals, boot_density, label="Бутстрап", linestyle="--")
plt.title("Плотность среднего: ЦПТ vs Бутстрап")
plt.xlabel("x")
plt.ylabel("Плотность")
plt.legend()
plt.show()

# (d) Бутстраповская оценка плотности коэффициента асимметрии и вероятность {асимметрия < 1}
boot_skews = np.array([stats.skew(np.random.choice(sample, size=n, replace=True), bias=False) for _ in range(B)])
x_skew = np.linspace(min(boot_skews)-0.2, max(boot_skews)+0.2, 200)
kde_skew = stats.gaussian_kde(boot_skews)
skew_density = kde_skew(x_skew)
prob_skew_less_1 = np.mean(boot_skews < 1)

plt.figure(figsize=(6,4))
plt.plot(x_skew, skew_density)
plt.axvline(1, color='red', linestyle="--", label="Коэфф. = 1")
plt.title("Бутстраповская плотность коэффициента асимметрии")
plt.xlabel("Коэфф. асимметрии")
plt.ylabel("Плотность")
plt.legend()
plt.show()

print("Пункт (d):")
print("Вероятность, что коэффициент асимметрии < 1:", prob_skew_less_1)

# (e) Сравнение плотности распределения медианы с бутстраповской оценкой
# Теоретическая плотность медианы (13-я порядковая статистика для n=25)
k = 13
C = special.comb(n, k-1)  # 25!/(12!*12!)
x_med = np.linspace(0, np.max(sample)*1.5, 300)
f_med_theor = C * (1 - np.exp(-x_med))**(k-1) * (np.exp(-x_med))**(n - k + 1) * np.exp(-x_med)

boot_medians = np.array([np.median(np.random.choice(sample, size=n, replace=True)) for _ in range(B)])
kde_median = stats.gaussian_kde(boot_medians)
boot_med_density = kde_median(x_med)

plt.figure(figsize=(6,4))
plt.plot(x_med, f_med_theor, label="Теоретическая плотность медианы")
plt.plot(x_med, boot_med_density, label="Бутстраповская оценка", linestyle="--")
plt.title("Плотность распределения медианы")
plt.xlabel("Медиана")
plt.ylabel("Плотность")
plt.legend()
plt.show()

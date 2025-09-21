import numpy as np
from scipy.stats import norm

theta_true = 5.0
n = 100
alpha = 0.05

np.random.seed(228)
u = np.random.rand(n)
X = (1 - u)**(1/(1 - theta_true))

# Метод максимального правдоподобия:
theta_hat_MLE = 1 + n / np.sum(np.log(X))

z = norm.ppf(1 - alpha/2)
se = (theta_hat_MLE - 1) / np.sqrt(n)

# Асимптотический доверительный интервал
theta_left = theta_hat_MLE - z * se
theta_right = theta_hat_MLE + z * se

# Асимптотический доверительный интервал для медианы
m_left = 2**(1/(theta_right - 1))
m_right = 2**(1/(theta_left - 1))

# Точное значение медианы
m_true = 2 ** (1 / (theta_true - 1) )

print("Истинное значение параметра theta: {:.4f}".format(theta_true))
print("Истинное значение медианы: {:.4f}".format(m_true))
print("МП-оценка параметра theta: {:.4f}".format(theta_hat_MLE))
print("Асимптотический доверительный интервал для theta: [{:.4f}, {:.4f}]".format(theta_left, theta_right))
print("Доверительный интервал для медианы: [{:.4f}, {:.4f}]".format(m_left, m_right))

B = 1000

# Параметрический бутстрап
theta_boot_par = np.empty(B)
for b in range(B):
    u_b = np.random.rand(n)
    X_b = (1 - u_b)**(1/(1 - theta_hat_MLE))
    theta_boot_par[b] = 1 + n / np.sum(np.log(X_b))

# Непараметрический бутстрап
theta_boot_nonpar = np.empty(B)
for b in range(B):
    sample_b = np.random.choice(X, size=n, replace=True)
    theta_boot_nonpar[b] = 1 + n / np.sum(np.log(sample_b))

left_par, right_par = np.percentile(theta_boot_par, [100*alpha/2, 100*(1 - alpha/2)])
left_nonpar, right_nonpar = np.percentile(theta_boot_nonpar, [100*alpha/2, 100*(1 - alpha/2)])

print("Параметрический бутстраповский доверительный интервал для theta: [{:.4f}, {:.4f}]".format(left_par, right_par))
print("Непараметрический бутстраповский доверительный интервал для theta: [{:.4f}, {:.4f}]".format(left_nonpar, right_nonpar))

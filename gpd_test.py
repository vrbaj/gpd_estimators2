from scipy.stats import genpareto
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")


sigma = 1
gamma = 0.1
mu = 0
sample_size = 100000
random_numbers = genpareto.rvs(gamma, loc=mu, scale=sigma, size=sample_size)
fig, ax = plt.subplots(1, 1)
ax.hist(random_numbers, density=True, histtype="stepfilled", alpha=0.2, label="rnds")
ax.legend(loc="best", frameon=False)

print("maximum: {}".format(max(random_numbers)))
print("minimum: {}".format(min(random_numbers)))


sample_mean = np.mean(random_numbers)
sample_mean_sq = np.mean(random_numbers ** 2)
sample_variance = np.var(random_numbers)
sample_variance_sq = np.var(random_numbers ** 2)
sigma_est = 0.5 * sample_mean * sample_mean_sq / (sample_mean_sq - sample_mean ** 2)  # OK
gamma_est = 0.5 - (sample_mean ** 2 / (2 * (sample_mean_sq - sample_mean ** 2))) # ok
print("sigma closed est: {}".format(sigma_est))
print("gamma closed est: {}".format(gamma_est))

sigma_moments = 0.5 * sample_mean * (sample_mean ** 2 / sample_variance + 1)
gamma_moments = -0.5 * (sample_mean ** 2 / sample_variance - 1)
print("sigma moments est:  {}".format(sigma_moments))
print("gamma moments est: {}".format(gamma_moments))

# print(sigma_moments - sigma_est)
# print(gamma_moments - gamma_est)
# print(1 - sigma/sigma_moments)

mle_fit = genpareto.fit(random_numbers)
print("gamma mle {}".format(mle_fit[0]))
print("mu mle {}".format(mle_fit[1]))
print("sigma mle {}".format(mle_fit[2]))
plt.show()
import numpy as np
import numpy.random as npr

def bootstrap(data, num_samples, statistic, alpha):
    """Returns bootstrap estimate of 100.0*(1-alpha) CI for statistic."""
    n = len(data)
    idx = npr.randint(0, n, (num_samples, n))
    samples = data[idx]
    stat = np.sort(statistic(samples, 1))
    return (stat[int((alpha/2.0)*num_samples)],
            stat[int((1-alpha/2.0)*num_samples)])

if __name__ == '__main__':
    # data of interest is bimodal and obviously not normal
    x = np.concatenate([npr.normal(3, 1, 100), npr.normal(6, 2, 200)])

    # find mean 95% CI and 100,000 bootstrap samples
    low, high = bootstrap(x, 100000, np.mean, 0.05)


import matplotlib.pyplot as plt

# Generate some example data
x = np.random.normal(0, 1, 1000)

# Calculate mean and confidence interval
mean = np.mean(x)
std = np.std(x)
n = len(x)
num_bootstrap_samples = 1000
bootstrap_means = [np.mean(np.random.choice(x, n)) for _ in range(num_bootstrap_samples)]
bootstrap_mean_ci = np.percentile(bootstrap_means, [2.5, 97.5])

# Plotting
plt.figure(figsize=(10, 4))

# Histogram
plt.subplot(121)
plt.hist(x, bins=50, histtype='step')
plt.title('Histogram of Data')

# Bootstrap Confidence Interval
plt.subplot(122)
plt.hist(bootstrap_means, bins=30, color='skyblue', edgecolor='black')
plt.axvline(bootstrap_mean_ci[0], color='red', linestyle='--', linewidth=2, label='95% CI')
plt.axvline(bootstrap_mean_ci[1], color='red', linestyle='--', linewidth=2)
plt.axvline(mean, color='green', linestyle='-', linewidth=2, label='Mean')
plt.legend()
plt.title('Bootstrap 95% CI for Mean')

plt.tight_layout()
#plt.savefig('examples/bootstrap.png')
plt.show()

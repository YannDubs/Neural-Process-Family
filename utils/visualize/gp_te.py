import matplotlib.pyplot as plt
import stheno
import numpy as np
from wbml import plot

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

np.random.seed(1111)
# Initialise GP
kernel = stheno.Matern52().stretch(0.25)
gp = stheno.GP(kernel)

# Initialise some data and grab a context set
x_all = np.linspace(-2., 2., 500)
y_all = gp(x_all).sample()

context_inds = np.random.randint(0, 400, 6)
xc, yc = x_all[context_inds], y_all[context_inds]

# get shifted context set
tau = 0.7
xc_shift, yc_shift = xc + tau, yc

# Plot original context set
plt.scatter(xc, yc, color='black')
plt.xlim(-2., 2.)
plt.ylim(-2.2, 2.5)
plot.tweak(legend=False)
plt.savefig('original-context')
plt.close()

# Plot shifted context set
plt.scatter(xc_shift, yc_shift, color='black')
plt.arrow(x=-1.5, y=0.75, dx=0.5, dy=0, width=0.05, color='black')
plt.text(x=-1.25, y=.85, s=r'$\tau$', fontsize=32)
plt.xlim(-2., 2.)
plt.ylim(-2.2, 2.5)
plot.tweak(legend=False)
plt.savefig('shifted-context')
plt.close()

# GP predictive with original Dc
post = gp | (xc, yc)
mean, lower, upper = post(x_all).marginals()
plt.plot(x_all, mean, color='tab:blue')
plt.fill_between(x_all, lower, upper, color='tab:blue', alpha=0.3)
plt.xlim(-2., 2.)
plt.ylim(-2.2, 2.5)
plot.tweak(legend=False)
plt.savefig('original-predictive')
plt.close()

# GP predictive with shifted Dc
post = gp | (xc_shift, yc_shift)
mean, lower, upper = post(x_all).marginals()
plt.plot(x_all, mean, color='tab:blue')
plt.fill_between(x_all, lower, upper, color='tab:blue', alpha=0.3)
plt.arrow(x=-1.5, y=0.75, dx=0.5, dy=0, width=0.05, color='black')
plt.text(x=-1.25, y=.85, s=r'$\tau$', fontsize=32)
plt.xlim(-2., 2.)
plt.ylim(-2.2, 2.5)
plot.tweak(legend=False)
plt.savefig('shifted-predictive')
plt.close()

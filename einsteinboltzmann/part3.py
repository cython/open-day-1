from __future__ import division

from utils import print_time_taken
import matplotlib.pyplot as plt
import os
import cPickle as pickle
import numpy as np

from evolution_mod import compute_einstein_boltzmann_grid

#7.56e-24
k = 7.56e-24
with print_time_taken('Solving...'):
    sol = compute_einstein_boltzmann_grid(n_x=1000, k_grid=np.array([k]))

x_grid = sol.x_grid

fig = plt.gcf()
fig.clear()
axs = [fig.add_subplot(2, 2, idx + 1) for idx in range(4)]
axsiter = iter(axs)
       
# delta plot
idx_k = 0
ax = axsiter.next()
ax.semilogy(x_grid, abs(sol.delta[idx_k,:]), ':k', label=r'$|\delta|$')
ax.semilogy(x_grid, abs(sol.delta_b[idx_k,:]), '-k', label=r'$|\delta_b|$')
ax.set_ylim(1e-3, 1e5)
ax.legend()

# v plot
ax = axsiter.next()
ax.semilogy(x_grid, abs(sol.v[idx_k,:]), ':k', label=r'$|v|$')
ax.semilogy(x_grid, abs(sol.v[idx_k,:]), '-k', label=r'$|v_b|$')
ax.set_ylim(1e-4, 1e2)
ax.legend()

# Phi, Psi plot
ax = axsiter.next()
ax.plot(x_grid, sol.Phi[idx_k,:], ':k', label=r'$\Phi$')
ax.plot(x_grid, -sol.Psi[idx_k,:], '-k', label=r'$-\Psi$')
ax.set_ylim(-.05, 1.05)
ax.legend(loc='upper right')

# Theta plot
ax = axsiter.next()
ax.semilogy(x_grid, abs(sol.Theta[0, idx_k, :]), ':k', label=r'$|\Theta_0|$')
ax.semilogy(x_grid, abs(sol.Theta[1, idx_k, :]), '-k', label=r'$|\Theta_1|$')
ax.legend(loc='upper right')
ax.set_ylim(1e-4, 10)

plt.show()

'''
@author: Mattia Micheletta Merlin
@date: 2024-05-20
@description: This script allows to visualize the eigenstates of the angular momentum operator,
commoly known as spherical harmonics, Y_lm(theta, phi). In particular, it displays the probability
density function rho(theta) = |Y_lm(theta, phi)|^2, in cartesian, polar, and spherical coordinates.
'''

from scipy.special import legendre
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Quantum numbers
l = 3
m = 1

# Legendre polynomial of degree l: P_l(x) = (1/2l!)(d^l/dx^l)(x^2-1)^l
P_l = legendre(l)
# Legendre function of the second kind of degree l and order m: P_lm(x) = (-1)^m(1-x^2)^(m/2)(d^m/dx^m)P_l(x)
P_lm = lambda x: (-1)**m * (1-x**2)**(m/2) * np.polyder(P_l, m)(x)
# Spherical harmonic of degree l and order m
Y_lm = lambda theta, phi: P_lm(np.cos(theta)) * np.exp(1j*m*phi) * np.sqrt((2*l+1)/(4*np.pi) * math.factorial(l-m)/math.factorial(l+m))
# Radial probability density function:
rho = lambda x: (P_lm(np.cos(x)))**2 * (2*l+1)*math.factorial(l-m)/(4*np.pi*math.factorial(l+m))



# Plot a bunch of stuff
fig = plt.figure(figsize=(20, 8))



# Plot in cartesian coordinates theta and rho(theta)
ax1 = fig.add_subplot(221)
x = np.linspace(0, np.pi, 100)
y = rho(x)
ax1.plot(x, y, color='black')
ax1.fill_between(x, y, color='gray', alpha=0.5)
ax1.set_xlabel(r'$\theta$')
ax1.set_ylabel(r'$\rho(\theta)$')



# Plot in polar coordinates with rho(theta) as distance from the center
ax4 = fig.add_subplot(222, polar=True)
ax4.set_theta_zero_location("N")
theta = np.linspace(0, 2*np.pi, 100)
rho_values = rho(theta)
ax4.plot(theta, rho_values, color='black')
ax4.fill(theta, rho_values, color='gray', alpha=0.5)
ax4.set_rticks([])  # Remove radial ticks
ax4.set_title(r'$\rho(\theta)$')



# Plot section of the sphere with color corresponding to rho(theta)
ax2 = fig.add_subplot(223, polar=True)
theta = np.linspace(0, 2*np.pi, 100)
radius = np.ones_like(theta)
color = rho(theta)
ax2.set_theta_zero_location("N")
sc = ax2.scatter(theta, radius, c=color, cmap='gray')
ax2.set_rticks([])  # Remove radial ticks
cbar = plt.colorbar(sc, ax=ax2, location='bottom')
cbar.set_label(r'$\rho(\theta)$')



# plot a bunch of points on a sphere, with color corresponding to rho(theta), where theta is the elevation angle
ax3 = fig.add_subplot(224, projection='3d')
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2*np.pi, 100)
theta, phi = np.meshgrid(theta, phi)
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)
colors = plt.cm.gray(rho(theta) / np.max(rho(theta)))
surf = ax3.plot_surface(x, y, z, facecolors=colors, rstride=1, cstride=1, antialiased=True, shade=False, cmap='gray')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.set_title(r'$\rho(x, y, z) = \rho(\theta)$')



# Show the plot
plt.show()

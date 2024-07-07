from scipy.special import legendre
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Spherical harmonic of degree l and order m
def Y_lm(theta, phi, l, m):
    # Legendre polynomial of degree l
    P_l = legendre(l)
    # Legendre function of the second kind of degree l and order m
    if (m>0):
        P_lm = lambda x: (-1)**m * (1-x**2)**(m/2) * np.polyder(P_l, m)(x)
    elif m<0:
        P_lm = lambda x: (1-x**2)**(-m/2) * np.polyder(P_l, -m)(x) * math.factorial(l+m)/math.factorial(l-m)
    return P_lm(np.cos(theta)) * np.exp(1j*m*phi) * np.sqrt((2*l+1)/(4*np.pi) * math.factorial(l-m)/math.factorial(l+m))



# Plot a bunch of stuff
fig = plt.figure(figsize=(20, 8))

p_x_orbital = lambda theta, phi: (Y_lm(theta, phi, 1, 1) - Y_lm(theta, phi, 1, -1)) / np.sqrt(2) * -1
p_y_orbital = lambda theta, phi: (Y_lm(theta, phi, 1, 1) + Y_lm(theta, phi, 1, -1)) / np.sqrt(2) * 1j

# Plot the absolute value of the spherical harmonic in 3D ( it doesnt depend on r)
ax5 = fig.add_subplot(111, projection='3d')
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2 * np.pi, 100)
theta, phi = np.meshgrid(theta, phi)
r = np.abs(p_y_orbital(theta, phi))
x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)
colors = plt.cm.viridis(r / np.max(r))  # Normalize for better visualization
surf = ax5.plot_surface(x, y, z, facecolors=colors, rstride=1, cstride=1, antialiased=True, shade=False)
ax5.set_xlabel('X')
ax5.set_ylabel('Y')
ax5.set_zlabel('Z')
ax5.set_title(r'$|Y_{lm}(\theta, \phi)|$')
ax5.set_aspect('equal')
ax5.set_title(r'$|Y_{lm}(\theta, \phi)|$')

plt.show()
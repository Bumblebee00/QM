'''
@author: Mattia Micheletta Merlin
@date: 2024-07-07
@description: This script allows to visualize the wave function of a electron in a hydrogen atom.
Is displayed a slice of the wave function in the XY plane, at height z.
Is displayed the imaginary and real part of the wave function, and the probability density.
The user can change the quantum numbers n, l, and m using text boxes.
'''

import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Slider
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import numpy as np
import scipy.special
from math import sqrt, factorial

# Constants and functions from the provided code
Z = 3
a = 0.529

# initial quantum numbers (the user can change these)
n = 1
l = 0
m = 0

xy_range = np.linspace(-1, 1, 80)
X, Y = np.meshgrid(xy_range, xy_range)

def calculate_arrays(z, n, l, m):
    k = Z / (a * n)
    D = sqrt((2*k)**3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))
    associated_lag_pol = scipy.special.genlaguerre(2*l + 1, n - l - 1)


    def Psi_nlm(r, theta, phi):
        x = scipy.special.sph_harm(m, l, phi, theta)
        return D * x * (2 * k * r)**l * np.exp(-k * r) * associated_lag_pol(2 * k * r)

    rho = np.zeros(X.shape)
    real = np.zeros(X.shape)
    imag = np.zeros(X.shape)

    for i in range(len(xy_range)):
        for j in range(len(xy_range)):
            x = X[i, j]
            y = Y[i, j]
            r = sqrt(x**2 + y**2 + z**2)
            theta = np.arccos(z/r)
            phi = np.arctan2(y, x)
            rho[i, j] = abs(Psi_nlm(r, theta, phi))**2
            real[i, j] = Psi_nlm(r, theta, phi).real
            imag[i, j] = Psi_nlm(r, theta, phi).imag
    return rho, real, imag

rho_values, real_values, imag_values = calculate_arrays(0.1, n, l, m)


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8), sharey=True, gridspec_kw={'wspace': 0.5})

cax1 = make_axes_locatable(ax1).append_axes("right", size="4%", pad="1%")
cax2 = make_axes_locatable(ax2).append_axes("right", size="4%", pad="1%")
cax3 = make_axes_locatable(ax3).append_axes("right", size="4%", pad="1%")

c = ax1.contourf(X, Y, rho_values, levels=100)
fig.colorbar(c, cax=cax1)
ax1.set_aspect('equal', 'box')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Probability Density')

c = ax2.contourf(X, Y, real_values, levels=100)
fig.colorbar(c, cax=cax2)
ax2.set_aspect('equal', 'box')
ax2.set_title('Real Part')

c = ax3.contourf(X, Y, imag_values, levels=100)
fig.colorbar(c, cax=cax3)
ax3.set_aspect('equal', 'box')
ax3.set_title('Imaginary Part')

axz = fig.add_axes([0.2, 0.8, 0.63, 0.0225])
z_slider = Slider(
    ax=axz,
    label="Z",
    valmin=-10,
    valmax=10,
    valinit=0,
    orientation="horizontal",

)

def smooth(x):
    return x/10 + (1 - np.exp(-x/10))*x*2

# The function to be called anytime a slider's value changes
def update(val):
    z = smooth(z_slider.val)
    z_slider.valtext.set_text(f'{z:.2f}')

    rho_values, real_values, imag_values = calculate_arrays(z, n, l, m)
    ax1.cla()  # Clear the previous contours
    cax1.cla()
    c = ax1.contourf(X, Y, rho_values, levels=100)
    fig.colorbar(c, cax=cax1)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title(f'Probability Density')

    ax2.cla()  # Clear the previous contours
    cax2.cla()
    c = ax2.contourf(X, Y, real_values, levels=100)
    fig.colorbar(c, cax=cax2)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title(f'Real Part')

    ax3.cla()  # Clear the previous contours
    cax3.cla()
    c = ax3.contourf(X, Y, imag_values, levels=100)
    fig.colorbar(c, cax=cax3)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title(f'Imaginary Part')

    fig.canvas.draw_idle()  # Redraw the figure to update the plot
    fig.canvas.update()
    fig.canvas.flush_events()


# register the update function with the slider
z_slider.on_changed(update)

# Callback functions to handle input
def submit_n(text):
    global n, l, m
    n = int(text)
    if n < 1:
        n = 1
        text_box_n.set_val(str(n))
    if l >= n:
        l = n - 1
        text_box_l.set_val(str(l))
    if m > l:
        m = l
        text_box_m.set_val(str(m))
    if m < -l:
        m = -l
        text_box_m.set_val(str(m))
    update(0)

def submit_l(text):
    global l, m
    l = int(text)
    if l < 0:
        l = 0
        text_box_l.set_val(str(l))
    if l >= n:
        l = n - 1
        text_box_l.set_val(str(l))
    if m > l:
        m = l
        text_box_m.set_val(str(m))
    if m < -l:
        m = -l
        text_box_m.set_val(str(m))
    update(0)

def submit_m(text):
    global m
    m = int(text)
    if m > l:
        m = l
        text_box_m.set_val(str(m))
    if m < -l:
        m = -l
        text_box_m.set_val(str(m))
    update(0)


axbox_n = plt.axes([0.3, 0.05, 0.1, 0.075])
text_box_n = TextBox(axbox_n, 'n: ', initial=n)
text_box_n.on_submit(submit_n)

axbox_l = plt.axes([0.45, 0.05, 0.1, 0.075])
text_box_l = TextBox(axbox_l, 'l: ', initial=l)
text_box_l.on_submit(submit_l)

axbox_m = plt.axes([0.6, 0.05, 0.1, 0.075])
text_box_m = TextBox(axbox_m, 'm: ', initial=m)
text_box_m.on_submit(submit_m)

plt.show()
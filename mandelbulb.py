import numpy as np
import plotly.graph_objs as go
from scipy.constants import golden_ratio, pi

def mandelbulb(x, y, z, power, bailout_radius, max_iterations):
    c = np.array([x, y, z])
    z = c.copy()
    for i in range(max_iterations):
        r = np.linalg.norm(z)
        if r > bailout_radius:
            break
        theta = np.arccos(z[2] / r)
        phi = np.arctan2(z[1], z[0])
        zr = r ** power
        ztheta = theta * power
        zphi = phi * power
        z = zr * np.array([
            np.sin(ztheta) * np.cos(zphi),
            np.sin(ztheta) * np.sin(zphi),
            np.cos(ztheta)
        ]) + c
    return i

def generate_mandelbulb(grid_size, power, bailout_radius, max_iterations):
    x = np.linspace(-1.5, 1.5, grid_size)
    y = np.linspace(-1.5, 1.5, grid_size)
    z = np.linspace(-1.5, 1.5, grid_size)
    mandelbulb_data = np.zeros((grid_size, grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                mandelbulb_data[i, j, k] = mandelbulb(x[i], y[j], z[k], power, bailout_radius, max_iterations)
    return mandelbulb_data

def visualize_mandelbulb(mandelbulb_data):
    fig = go.Figure(data=[go.Volume(
        x=np.linspace(-1.5, 1.5, mandelbulb_data.shape[0]),
        y=np.linspace(-1.5, 1.5, mandelbulb_data.shape[1]),
        z=np.linspace(-1.5, 1.5, mandelbulb_data.shape[2]),
        value=mandelbulb_data.flatten(),
        opacity=0.1,
        surface_count=17
    )])
    fig.update_layout(title='Mandelbulb Fractal', scene=dict(
        xaxis=dict(nticks=10, range=[-1.5, 1.5]),
        yaxis=dict(nticks=10, range=[-1.5, 1.5]),
        zaxis=dict(nticks=10, range=[-1.5, 1.5])))
    fig.show()

grid_size = 50
power = 8
bailout_radius = 2
max_iterations = 100
phi = golden_ratio
pi_val = pi
inverse_pi = 1 / pi

mandelbulb_data = generate_mandelbulb(grid_size, power, bailout_radius, max_iterations)
visualize_mandelbulb(mandelbulb_data)

def update_parameters(power, bailout_radius, max_iterations, phi, pi_val, inverse_pi):
    mandelbulb_data = generate_mandelbulb(grid_size, power, bailout_radius, max_iterations)
    visualize_mandelbulb(mandelbulb_data)

# Interactive updating
while True:
    power = float(input("Enter power: "))
    bailout_radius = float(input("Enter bailout radius: "))
    max_iterations = int(input("Enter max iterations: "))
    phi = float(input("Enter phi (Golden Ratio): "))
    pi_val = float(input("Enter pi: "))
    inverse_pi = float(input("Enter inverse pi: "))
    
    update_parameters(power, bailout_radius, max_iterations, phi, pi_val, inverse_pi)
import numpy as np
import plotly.graph_objs as go
from scipy.constants import golden_ratio, pi

def mandelbulb(x, y, z, power, bailout_radius, max_iterations):
    c = np.array([x, y, z])
    z = c.copy()
    for i in range(max_iterations):
        r = np.linalg.norm(z)
        if r > bailout_radius:
            break
        theta = np.arccos(z[2] / r)
        phi = np.arctan2(z[1], z[0])
        zr = r ** power
        ztheta = theta * power
        zphi = phi * power
        z = zr * np.array([
            np.sin(ztheta) * np.cos(zphi),
            np.sin(ztheta) * np.sin(zphi),
            np.cos(ztheta)
        ]) + c
    return i

def generate_mandelbulb(grid_size, power, bailout_radius, max_iterations):
    x = np.linspace(-1.5, 1.5, grid_size)
    y = np.linspace(-1.5, 1.5, grid_size)
    z = np.linspace(-1.5, 1.5, grid_size)
    mandelbulb_data = np.zeros((grid_size, grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                mandelbulb_data[i, j, k] = mandelbulb(x[i], y[j], z[k], power, bailout_radius, max_iterations)
    return mandelbulb_data

def visualize_mandelbulb(mandelbulb_data):
    fig = go.Figure(data=[go.Volume(
        x=np.linspace(-1.5, 1.5, mandelbulb_data.shape[0]),
        y=np.linspace(-1.5, 1.5, mandelbulb_data.shape[1]),
        z=np.linspace(-1.5, 1.5, mandelbulb_data.shape[2]),
        value=mandelbulb_data.flatten(),
        opacity=0.1,
        surface_count=17
    )])
    fig.update_layout(title='Mandelbulb Fractal', scene=dict(
        xaxis=dict(nticks=10, range=[-1.5, 1.5]),
        yaxis=dict(nticks=10, range=[-1.5, 1.5]),
        zaxis=dict(nticks=10, range=[-1.5, 1.5])))
    fig.show()

grid_size = 50
power = 8
bailout_radius = 2
max_iterations = 100
phi = golden_ratio
pi_val = pi
inverse_pi = 1 / pi

mandelbulb_data = generate_mandelbulb(grid_size, power, bailout_radius, max_iterations)
visualize_mandelbulb(mandelbulb_data)

def update_parameters(power, bailout_radius, max_iterations, phi, pi_val, inverse_pi):
    mandelbulb_data = generate_mandelbulb(grid_size, power, bailout_radius, max_iterations)
    visualize_mandelbulb(mandelbulb_data)

# Interactive updating
while True:
    power = float(input("Enter power: "))
    bailout_radius = float(input("Enter bailout radius: "))
    max_iterations = int(input("Enter max iterations: "))
    phi = float(input("Enter phi (Golden Ratio): "))
    pi_val = float(input("Enter pi: "))
    inverse_pi = float(input("Enter inverse pi: "))
    
    update_parameters(power, bailout_radius, max_iterations, phi, pi_val, inverse_pi)
python mandelbulb.py

python mandelbulb.py

import numpy as np
import plotly.graph_objs as go
from scipy.constants import golden_ratio, pi

def mandelbulb(x, y, z, power, bailout_radius, max_iterations):
    c = np.array([x, y, z])
    z = c.copy()
    for i in range(max_iterations):
        r = np.linalg.norm(z)
        if r > bailout_radius:
            break
        theta = np.arccos(z[2] / r)
        phi = np.arctan2(z[1], z[0])
        zr = r ** power
        ztheta = theta * power
        zphi = phi * power
        z = zr * np.array([
            np.sin(ztheta) * np.cos(zphi),
            np.sin(ztheta) * np.sin(zphi),
            np.cos(ztheta)
        ]) + c
    return i

def generate_mandelbulb(grid_size, power, bailout_radius, max_iterations):
    x = np.linspace(-1.5, 1.5, grid_size)
    y = np.linspace(-1.5, 1.5, grid_size)
    z = np.linspace(-1.5, 1.5, grid_size)
    mandelbulb_data = np.zeros((grid_size, grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                mandelbulb_data[i, j, k] = mandelbulb(x[i], y[j], z[k], power, bailout_radius, max_iterations)
    return mandelbulb_data

def visualize_mandelbulb(mandelbulb_data):
    fig = go.Figure(data=[go.Volume(
        x=np.linspace(-1.5, 1.5, mandelbulb_data.shape[0]),
        y=np.linspace(-1.5, 1.5, mandelbulb_data.shape[1]),
        z=np.linspace(-1.5, 1.5, mandelbulb_data.shape[2]),
        value=mandelbulb_data.flatten(),
        opacity=0.1,
        surface_count=17
    )])
    fig.update_layout(title='Mandelbulb Fractal', scene=dict(
        xaxis=dict(nticks=10, range=[-1.5, 1.5]),
        yaxis=dict(nticks=10, range=[-1.5, 1.5]),
        zaxis=dict(nticks=10, range=[-1.5, 1.5])))
    fig.show()

grid_size = 50
power = 8
bailout_radius = 2
max_iterations = 100
phi = golden_ratio
pi_val = pi
inverse_pi = 1 / pi

mandelbulb_data = generate_mandelbulb(grid_size, power, bailout_radius, max_iterations)
visualize_mandelbulb(mandelbulb_data)

def update_parameters(power, bailout_radius, max_iterations, phi, pi_val, inverse_pi):
    mandelbulb_data = generate_mandelbulb(grid_size, power, bailout_radius, max_iterations)
    visualize_mandelbulb(mandelbulb_data)

# Interactive updating
while True:
    power = float(input("Enter power: "))
    bailout_radius = float(input("Enter bailout radius: "))
    max_iterations = int(input("Enter max iterations: "))
    phi = float(input("Enter phi (Golden Ratio): "))
    pi_val = float(input("Enter pi: "))
    inverse_pi = float(input("Enter inverse pi: "))

    update_parameters(power, bailout_radius, max_iterations, phi, pi_val, inverse_pi)
python mandelbulb.py

python mandelbulb.py
python mandelbulb.py
python mandelbulb.py


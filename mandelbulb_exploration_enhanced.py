import numpy as np
import plotly.graph_objs as go
from scipy.constants import golden_ratio, pi
import streamlit as st
import matplotlib.pyplot as plt

# Mandelbulb generation function
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

# Function to generate Mandelbulb fractal data
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

# Function to visualize Mandelbulb fractal data
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

# Function to update parameters and regenerate Mandelbulb data
def update_parameters(power, bailout_radius, max_iterations):
    mandelbulb_data = generate_mandelbulb(grid_size, power, bailout_radius, max_iterations)
    visualize_mandelbulb(mandelbulb_data)

# Function to perform 1D golden section search
def search_1d(a, b, f, tol=1e-5):
    c = b - (b - a) / golden_ratio
    d = a + (b - a) / golden_ratio
    while abs(b - a) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c
        c = b - (b - a) / golden_ratio
        d = a + (b - a) / golden_ratio
    return (b + a) / 2

# Function to perform golden section search in 3D
def golden_section_search(x_min, x_max, y_min, y_max, z_min, z_max, power, bailout_radius, max_iterations, resolution):
    segment_size_x = (x_max - x_min) / golden_ratio
    segment_size_y = (y_max - y_min) / golden_ratio
    segment_size_z = (z_max - z_min) / golden_ratio
    
    mid_x = x_min + segment_size_x
    mid_y = y_min + segment_size_y
    mid_z = z_min + segment_size_z
    
    points = []
    values = []
    
    def eval_point(x, y, z):
        value = mandelbulb(x, y, z, power, bailout_radius, max_iterations)
        points.append((x, y, z))
        values.append(value)
        return value
    
    eval_point(mid_x, mid_y, mid_z)
    
    if resolution > 1:
        golden_section_search(x_min, mid_x, y_min, mid_y, z_min, mid_z, power, bailout_radius, max_iterations, resolution - 1)
        golden_section_search(mid_x, x_max, mid_y, y_max, mid_z, z_max, power, bailout_radius, max_iterations, resolution - 1)
    
    return points, values

# Function to save a 2D slice of the Mandelbulb data as an image
def save_mandelbulb_image(mandelbulb_data, filename="mandelbulb.png"):
    plt.imshow(mandelbulb_data[:, :, mandelbulb_data.shape[2] // 2], cmap='inferno')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

# Streamlit app
st.title("Mandelbulb Exploration with Golden Section Search")

# Parameters
power = st.sidebar.slider("Power", 2, 10, 8)
bailout_radius = st.sidebar.slider("Bailout Radius", 1, 10, 2)
max_iterations = st.sidebar.slider("Max Iterations", 50, 500, 100)
x_min = st.sidebar.slider("X Min", -10.0, 0.0, -2.0)
x_max = st.sidebar.slider("X Max", 0.0, 10.0, 2.0)
y_min = st.sidebar.slider("Y Min", -10.0, 0.0, -2.0)
y_max = st.sidebar.slider("Y Max", 0.0, 10.0, 2.0)
z_min = st.sidebar.slider("Z Min", -10.0, 0.0, -2.0)
z_max = st.sidebar.slider("Z Max", 0.0, 10.0, 2.0)
resolution = st.sidebar.slider("Resolution", 1, 50, 10)

# Start exploration
points, values = golden_section_search(x_min, x_max, y_min, y_max, z_min, z_max, power, bailout_radius, max_iterations, resolution)

# Plot points
fig, ax = plt.subplots()
sc = ax.scatter([p[0] for p in points], [p[1] for p in points], c=values, cmap='viridis')
plt.colorbar(sc, label='Mandelbulb Value')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Mandelbulb Exploration')

st.pyplot(fig)

# Log points
log_data = np.array(points)
np.save('log_data.npy', log_data)

# Provide feedback on computation status
st.sidebar.write(f"Generated {len(points)} points")

# Button to save the Mandelbulb image
if st.sidebar.button("Save Mandelbulb Image"):
    mandelbulb_data = generate_mandelbulb(grid_size, power, bailout_radius, max_iterations)
    save_mandelbulb_image(mandelbulb_data)
    st.sidebar.write("Image saved as mandelbulb.png")

# Main execution
if __name__ == '__main__':
    grid_size = 50
    power = 8
    bailout_radius = 2
    max_iterations = 100

    mandelbulb_data = generate_mandelbulb(grid_size, power, bailout_radius, max_iterations)
    visualize_mandelbulb(mandelbulb_data)
    
    while True:
        power = float(input("Enter power: "))
        bailout_radius = float(input("Enter bailout radius: "))
        max_iterations = int(input("Enter max iterations: "))
        update_parameters(power, bailout_radius, max_iterations)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st
from deap import base, creator, tools, algorithms  # import DEAP for evolutionary algorithm
import requests
import json
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc

# Define the constants
pi = np.pi
phi = (1 + np.sqrt(5)) / 2
inv_pi = 1 / pi

# Mandelbulb generation functions (taken from previous code)
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

def save_mandelbulb_image(mandelbulb_data, filename="mandelbulb.png"):
    plt.imshow(mandelbulb_data[:, :, mandelbulb_data.shape[2] // 2], cmap='inferno')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def visualize_mandelbulb(mandelbulb_data):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(mandelbulb_data, edgecolor='k')
    ax.set_title('Mandelbulb Fractal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    st.pyplot(fig)

# Function to create a module with metadata in Clarifai (taken from previous code)
def create_module_with_metadata(auth_key, metadata):
    headers = {
        'Authorization': f'Key {auth_key}',
        'Content-Type': 'application/json'
    }
    url = "https://api.clarifai.com/v2/modules"
    response = requests.post(url, headers=headers, data=json.dumps(metadata))
    return response

def render_create_module(auth_key, metadata):
    st.write("Metadata JSON:", json.dumps(metadata))  # Debugging line to check JSON content

    # Create module with metadata
    try:
        response = create_module_with_metadata(auth_key, metadata)
        if response.status_code == 200:
            st.success("Module created successfully!")
        else:
            st.error(f"Failed to create module: {response.content}")
    except Exception as e:
        st.error(f"Error while creating module: {str(e)}")

def predict_with_clarifai(auth_key):
    st.subheader("Clarifai Model Prediction")

    # Input parameters for Clarifai prediction
    model_id = "Proto-Labr-Nth-Guide"
    image_url = st.text_input("Image URL")
    api_key = auth_key  # Use the same API key

    if st.button("Predict"):
        # Setup gRPC channel
        channel = ClarifaiChannel.get_grpc_channel()
        stub = service_pb2_grpc.V2Stub(channel)

        # Prepare request
        request = service_pb2.PostModelOutputsRequest(
            model_id=model_id,
            inputs=[
                resources_pb2.Input(data=resources_pb2.Data(image=resources_pb2.Image(url=image_url)))
            ]
        )

        metadata = (('authorization', f'Key {api_key}'),)

        try:
            # Make prediction request
            response = stub.PostModelOutputs(request, metadata=metadata)
            st.write("Model prediction:", response)
        except Exception as e:
            st.error(f"Error during model prediction: {e}")

# Define the evolutionary algorithm
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, -1, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=6)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", lambda x: fractal_function(x[0], x[1], x[2], x[3], x[4], x[5]))
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

def evolutionary_algorithm(population, toolbox, ngen, stats):
    for gen in range(ngen):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
        fits = [toolbox.evaluate(ind) for ind in offspring]
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
        stats.update(population)
    return population

# Function to evaluate individuals in evolutionary algorithm (taken from previous code)
def fractal_function(a, b, c, d, e, f):
    return (a * pi + b * phi + c * inv_pi) + (d * phi + e * inv_pi + f * pi)

# Function to generate a tesseract for visualization
def generate_tesseract():
    n_dim = 4
    n_cells = 2
    tesseract = np.zeros((n_cells ** n_dim, n_dim))
    for i in range(n_cells ** n_dim):
        cell = np.zeros((n_dim,))
        for j in range(n_dim):
            cell[j] = (i // (n_cells ** j)) % n_cells
        tesseract[i] = cell
    return tesseract

# Function to project the tesseract for visualization
def project_tesseract(tesseract):
    # Project the 4D tesseract to 3D
    projected_tesseract = np.zeros((tesseract.shape[0], 3))
    for i in range(tesseract.shape[0]):
        projected_tesseract[i, 0] = tesseract[i, 0] - tesseract[i, 1]
        projected_tesseract[i, 1] = tesseract[i, 2] + tesseract[i, 3]
        projected_tesseract[i, 2] = tesseract[i, 0] + tesseract[i, 1]
    return projected_tesseract

# Function to visualize the projected tesseract
def visualize_tesseract(projected_tesseract):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(projected_tesseract[:, 0], projected_tesseract[:, 1], projected_tesseract[:, 2])
    ax.set_title('Tesseract Projection')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    st.pyplot(fig)

# Function to generate a fractal image (placeholder, replace with actual implementation)
def generate_fractal():
    # Replace with your fractal generation logic
    pass

# Main function to run the application
def main():
    st.title("Mandelbulb Generator and Clarifai Module Creator")

    # Sidebar inputs for Mandelbulb generation (example)
    grid_size = st.sidebar.number_input("Grid Size", min_value=10, max_value=100, value=50)
    power = st.sidebar.number_input("Power", min_value=2, max_value=10, value=8)
    bailout_radius = st.sidebar.number_input("Bailout Radius", min_value=1.0, max_value=10.0, value=2.0)
    max_iterations = st.sidebar.number_input("Max Iterations", min_value=10, max_value=1000, value=100)

    # Generate and visualize Mandelbulb (example)
    if st.sidebar.button("Generate Mandelbulb"):
        mandelbulb_data = generate_mandelbulb(grid_size, power, bailout_radius, max_iterations)
        visualize_mandelbulb(mandelbulb_data)
        save_mandelbulb_image(mandelbulb_data)

    # Evolutionary Algorithm Section (example)
    st.header("Evolutionary Algorithm Example")

    population_size = st.slider("Population Size", min_value=10, max_value=100, value=50)
    num_generations = st.slider("Number of Generations", min_value=10, max_value=100, value=40)

    # Run evolutionary algorithm
    if st.button("Run Evolutionary Algorithm"):
        population = toolbox.population(n=population_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        population = evolutionary_algorithm(population, toolbox, num_generations, stats)
        best_individual = hof[0]
        st.write("Best Individual:", best_individual)

    # Tesseract Visualization Section (example)
    st.header("Tesseract Visualization Example")

    if st.button("Visualize Tesseract"):
        tesseract = generate_tesseract()
        projected_tesseract = project_tesseract(tesseract)
        visualize_tesseract(projected_tesseract)

    # Clarifai Section (example)
    st.header("Clarifai Integration Example")

    auth_key = st.text_input("Clarifai API Key")
    if st.button("Create Module with Metadata"):
        metadata = {"name": "My Module", "description": "Custom module for image recognition"}
        render_create_module(auth_key, metadata)

    if st.button("Predict with Clarifai"):
        predict_with_clarifai(auth_key)

if __name__ == "__main__":
    main()






    

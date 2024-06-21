import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st
from deap import base, creator, tools, algorithms
import requests
import json
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
import os

# Constants
pi = np.pi
phi = (1 + np.sqrt(5)) / 2
inv_pi = 1 / pi

# Create the directory to save plots
os.makedirs('plots', exist_ok=True)

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

def create_module_with_metadata(auth_key, metadata):
    headers = {
        'Authorization': f'Key {auth_key}',
        'Content-Type': 'application/json'
    }
    url = "https://api.clarifai.com/v2/modules"
    response = requests.post(url, headers=headers, data=json.dumps(metadata))
    return response

def render_create_module(auth_key, metadata):
    st.write("Metadata JSON:", json.dumps(metadata))
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
    model_id = "Proto-Labr-Nth-Guide"
    image_url = st.text_input("Image URL")
    api_key = auth_key

    if st.button("Predict"):
        channel = ClarifaiChannel.get_grpc_channel()
        stub = service_pb2_grpc.V2Stub(channel)
        request = service_pb2.PostModelOutputsRequest(
            model_id=model_id,
            inputs=[
                resources_pb2.Input(data=resources_pb2.Data(image=resources_pb2.Image(url=image_url)))
            ]
        )
        metadata = (('authorization', f'Key {api_key}'),)

        try:
            response = stub.PostModelOutputs(request, metadata=metadata)
            if response.status.code == 10000:
                st.write("Predictions:")
                for concept in response.outputs[0].data.concepts:
                    st.write(f"{concept.name}: {concept.value:.2f}")
            else:
                st.error(f"Failed to get predictions: {response.status.description}")
        except Exception as e:
            st.error(f"Error during model prediction: {e}")

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
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    population = toolbox.population(n=len(population))
    hof = tools.HallOfFame(1)

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    hof.update(population)
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(population), **record)

    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring
        hof.update(population)
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

    return population, logbook, hof

def fractal_function(a, b, c, d, e, f):
    return (a * pi + b * phi + c * inv_pi) + (d * phi + e * inv_pi + f * pi)

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

def project_tesseract(tesseract):
    projected_tesseract = np.zeros((tesseract.shape[0], 3))
    for i in range(tesseract.shape[0]):
        projected_tesseract[i, 0] = tesseract[i, 0] - tesseract[i, 1]
        projected_tesseract[i, 1] = tesseract[i, 2] + tesseract[i, 3]
        projected_tesseract[i, 2] = tesseract[i, 0] + tesseract[i, 1]
    return projected_tesseract

def visualize_tesseract(projected_tesseract):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(projected_tesseract[:, 0], projected_tesseract[:, 1], projected_tesseract[:, 2])
    ax.set_title('Tesseract Projection')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    st.pyplot(fig)

def main():
    st.title("Fractal and Tesseract Visualization with Evolutionary Algorithm")

    st.header("Mandelbulb Fractal Example")
    grid_size = st.slider("Grid Size", min_value=10, max_value=100, value=30)
    power = st.slider("Power", min_value=2, max_value=8, value=8)
    bailout_radius = st.slider("Bailout Radius", min_value=2, max_value=20, value=10)
    max_iterations = st.slider("Max Iterations", min_value=10, max_value=100, value=20)
    
    if st.button("Generate Mandelbulb"):
        mandelbulb_data = generate_mandelbulb(grid_size, power, bailout_radius, max_iterations)
        visualize_mandelbulb(mandelbulb_data)
        save_mandelbulb_image(mandelbulb_data)

    st.header("Evolutionary Algorithm Example")
    population_size = st.slider("Population Size", min_value=10, max_value=100, value=50)
    num_generations = st.slider("Number of Generations", min_value=10, max_value=100, value=50)
    
    if st.button("Run Evolutionary Algorithm"):
        population = toolbox.population(n=population_size)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        population, logbook, hof = evolutionary_algorithm(population, toolbox, num_generations, stats)
        best_individual = hof[0]
        st.write("Best Individual:", best_individual)

    st.header("Tesseract Visualization Example")
    
    if st.button("Visualize Tesseract"):
        tesseract = generate_tesseract()
        projected_tesseract = project_tesseract(tesseract)
        visualize_tesseract(projected_tesseract)

    st.header("Clarifai Integration Example")

    auth_key = st.text_input("92186bbf0c584e378fea53af41f855b3")  
    if st.button("Create Module with Metadata"):
        metadata = {"name": "Module_PLN", "description": "Custom module for image recognition and Mandelbulb operation"}
        render_create_module(auth_key, metadata)

    if st.button("Predict with Clarifai"):
        predict_with_clarifai(auth_key)

if __name__ == "__main__":
    main()


    auth_key = st.text_input("92186bbf0c584e378fea53af41f855b3")  
    if st.button("Create Module with Metadata"):
        metadata = {"name": "Module_PLN", "description": "Custom module for image recognition and Mandelbulb operation"}
        render_create_module(auth_key, metadata)

    if st.button("Predict with Clarifai"):
        predict_with_clarifai(auth_key)

if __name__ == "__main__":
    main()






    

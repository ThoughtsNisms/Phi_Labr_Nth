import os
import requests
import json
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_pb2, status_code_pb2
import streamlit as st

# Mandelbulb generation functions
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

# Function to create a module with metadata in Clarifai
def create_module_with_metadata(auth_key, metadata):
    headers = {
        'Authorization': f'Key {auth_key}',
        'Content-Type': 'application/json'
    }
    url = "https://api.clarifai.com/v2/modules"
    response = requests.post(url, headers=headers, data=metadata)
    return response

def render_create_module(auth_key):
    # Define metadata for the module
    metadata = {
        "name": "Phi-Labr^Nth",
        "description": ("An advanced AI application that immerses users in the intricate virtual world of Mandelbulb fractals. "
                        "The core functionality revolves around generating, exploring, and manipulating 3D structures within the Mandelbulb space "
                        "under varying environmental parameters. This exploration leverages the mathematical constants Phi (the golden ratio), Pi, "
                        "and the inverse of Pi to simulate organic formations, pressure maintenance, and structural fracturing, respectively."),
        "version": "1.0.0",
        "author": "Your Name",
        "license": "MIT",
        "keywords": [
            "AI",
            "Mandelbulb",
            "Fractals",
            "3D Visualization",
            "Mathematical Constants",
            "Phi",
            "Pi",
            "Inverse Pi"
        ],
        "dependencies": {
            "clarifai": "9.8.1",
            "streamlit": "1.24.0",
            "numpy": "*",
            "matplotlib": "*",
            "plotly": "*",
            "clarifai-grpc": "*"
        },
        "main_file": "mandelbulb_clarifai.py",
        "entry_points": {
            "main": "mandelbulb_clarifai:main"
        }
    }

    # Convert metadata to JSON string
    metadata_json = json.dumps(metadata)
    st.write("Metadata JSON:", metadata_json)  # Debugging line to check JSON content

    # Create module with metadata
    try:
        response = create_module_with_metadata(auth_key, metadata_json)
        if response.status_code == 200:
            st.success("Module created successfully!")
        else:
            st.error(f"Failed to create module: {response.content}")
    except Exception as e:
        st.error(f"Error while creating module: {str(e)}")

def main():
    st.title("Mandelbulb Generator and Clarifai Module Creator")

    # Sidebar inputs for Mandelbulb generation
    grid_size = st.sidebar.number_input("Grid Size", min_value=10, max_value=100, value=50)
    power = st.sidebar.number_input("Power", min_value=2, max_value=10, value=8)
    bailout_radius = st.sidebar.number_input("Bailout Radius", min_value=1.0, max_value=10.0, value=2.0)
    max_iterations = st.sidebar.number_input("Max Iterations", min_value=10, max_value=1000, value=100)

    # Generate and visualize Mandelbulb
    if st.sidebar.button("Generate Mandelbulb"):
        mandelbulb_data = generate_mandelbulb(grid_size, power, bailout_radius, max_iterations)
        save_mandelbulb_image(mandelbulb_data, filename="mandelbulb.png")
        visualize_mandelbulb(mandelbulb_data)
        st.image("mandelbulb.png", caption="Generated Mandelbulb")

    # Form for Clarifai module creation
    with st.form(key="module_form"):
        st.write("Clarifai Module Metadata")
        name = st.text_input("Module Name", value="Phi-Labr^Nth")
        description = st.text_area("Description", value=(
            "An advanced AI application that immerses users in the intricate virtual world of Mandelbulb fractals. "
            "The core functionality revolves around generating, exploring, and manipulating 3D structures within the Mandelbulb space "
            "under varying environmental parameters. This exploration leverages the mathematical constants Phi (the golden ratio), Pi, "
            "and the inverse of Pi to simulate organic formations, pressure maintenance, and structural fracturing, respectively."))
        version = st.text_input("Version", value="1.0.0")
        author = st.text_input("Author", value="Your Name")
        license = st.text_input("License", value="MIT")
        keywords = st.text_area("Keywords", value="AI, Mandelbulb, Fractals, 3D Visualization, Mathematical Constants, Phi, Pi, Inverse Pi")
        
        # Clarifai API Key
        auth_key = st.text_input("Clarifai API Key", type="password")
        
        # Submit button
        submit_button = st.form_submit_button(label="Create Module")
        
        if submit_button:
            metadata = {
                "name": name,
                "description": description,
                "version": version,
                "author": author,
                "license": license,
                "keywords": keywords.split(", "),
                "dependencies": {
                    "clarifai": "9.8.1",
                    "streamlit": "1.24.0",
                    "numpy": "*",
                    "matplotlib": "*",
                    "plotly": "*",
                    "clarifai-grpc": "*"
                },
                "main_file": "mandelbulb_clarifai.py",
                "entry_points": {
                    "main": "mandelbulb_clarifai:main"
                }
            }
            
            # Convert metadata to JSON string
            metadata_json = json.dumps(metadata)
            st.write("Metadata JSON:", metadata_json)  # Debugging line to check JSON content
            
            # Create module with metadata
            render_create_module(auth_key)
    
    # Clarifai model prediction using gRPC
    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)
    
    model_id = "Proto-Labr-Nth-Guide"
    image_url = "https://s3.amazonaws.com/samples.clarifai.com/featured-models/image-captioning-statue-of-liberty.jpeg"
    api_key = auth_key  # Use the same API key
    
    request = service_pb2.PostModelOutputsRequest(
        model_id=model_id,
        inputs=[
            resources_pb2.Input(data=resources_pb2.Data(image=resources_pb2.Image(url=image_url)))
        ]
    )
    
    metadata = (('authorization', f'Key {api_key}'),)
    
    try:
        response = stub.PostModelOutputs(request, metadata=metadata)
        st.write("Model prediction:", response)
    except Exception as e:
        st.error(f"Error during model prediction: {e}")

if __name__ == "__main__":
    main()



    

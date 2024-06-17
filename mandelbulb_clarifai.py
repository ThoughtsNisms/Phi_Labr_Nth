import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_pb2, status_code_pb2

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

def main():
    # Constants for Mandelbulb
    grid_size = 50
    power = 8
    bailout_radius = 2
    max_iterations = 100

    # Generate and visualize Mandelbulb
    mandelbulb_data = generate_mandelbulb(grid_size, power, bailout_radius, max_iterations)
    save_mandelbulb_image(mandelbulb_data, filename="mandelbulb.png")
    visualize_mandelbulb(mandelbulb_data)
    
    # Clarifai model prediction using gRPC
    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)
    
    model_url = "https://clarifai.com/ckqhro0evz4c/Phi-Labrnth/models/Proto-Labr-Nth-Guide"
    image_url = "https://s3.amazonaws.com/samples.clarifai.com/featured-models/image-captioning-statue-of-liberty.jpeg"
    api_key = "92186bbf0c584e378fea53af41f855b3"
    
    request = service_pb2.PostModelOutputsRequest(
        model_id=model_url,
        inputs=[
            resources_pb2.Input(data=resources_pb2.Data(image=resources_pb2.Image(url=image_url)))
        ]
    )
    
    metadata = (('authorization', f'Key {api_key}'),)
    
    try:
        response = stub.PostModelOutputs(request, metadata=metadata)
        print("Model prediction:", response)
    except Exception as e:
        print(f"Error during model prediction: {e}")

if __name__ == "__main__":
    main()


    

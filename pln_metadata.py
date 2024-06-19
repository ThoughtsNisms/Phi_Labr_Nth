import json

metadata = {
    "name": "Phi-Labr^Nth",
    "description": (
        "Phi-Labr^Nth is an advanced AI application designed to immerse users in the intricate virtual world of Mandelbulb fractals. "
        "The core functionality involves generating, exploring, and manipulating 3D structures within the Mandelbulb space under varying "
        "environmental parameters. This exploration leverages mathematical constants such as Phi (the golden ratio), Pi, and the inverse of Pi "
        "to simulate organic formations, pressure maintenance, and structural fracturing, respectively. The application integrates state-of-the-art "
        "visualization tools to provide an interactive and educational experience for users."
    ),
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
        "Inverse Pi",
        "Education",
        "Interactive"
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

with open('module_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

print("Metadata saved to module_metadata.json")


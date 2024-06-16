import mandelbulb_generation
import organic_formations
import structural_fracturing

def get_user_input():
    """
    Get user input for Phi, Pi, and inverse Pi with input validation.
    """
    while True:
        try:
            phi = float(input("Enter value for Phi (default: 1.61803398875): ") or 1.61803398875)
            pi = float(input("Enter value for Pi (default: 3.14159265359): ") or 3.14159265359)
            inverse_pi = 1 / pi
            grid_size = int(input("Enter grid size (default: 100): ") or 100)
            power = int(input("Enter power (default: 8): ") or 8)
            bailout_radius = float(input("Enter bailout radius (default: 2): ") or 2)
            max_iterations = int(input("Enter max iterations (default: 100): ") or 100)
            break
        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")
    return phi, pi, inverse_pi, grid_size, power, bailout_radius, max_iterations

def main():
    """
    Main function to generate Mandelbulb fractal data, simulate organic formations, 
    simulate structural fracturing, and visualize the results.
    """
    phi, pi, inverse_pi, grid_size, power, bailout_radius, max_iterations = get_user_input()

    # Generate Mandelbulb structure
    mandelbulb_structure = mandelbulb_generation.generate_mandelbulb_structure(
        phi, pi, inverse_pi, grid_size, power, bailout_radius, max_iterations
    )

    # Simulate organic formations
    environmental_parameters = {'temperature': 20, 'humidity': 0.5}
    organic_structure = organic_formations.simulate_organic_formations(mandelbulb_structure, environmental_parameters)

    # Simulate structural fracturing
    pressure_parameters = {'pressure': 10, 'stress': 0.1}
    fractured_structure = structural_fracturing.simulate_structural_fracturing(organic_structure, pressure_parameters)

    # Save the modified Mandelbulb structure as an image
    mandelbulb_generation.save_mandelbulb_image(fractured_structure, filename="modified_mandelbulb.png")

    # Visualize the Mandelbulb structure
    mandelbulb_generation.visualize_mandelbulb(fractured_structure)

    # Upload the generated Mandelbulb image to Clarifai
    user_id = "YOUR_USER_ID"  # Replace with your Clarifai User ID
    pat_key = "YOUR_PAT_KEY"  # Replace with your Clarifai Personal Access Token
    app_id = "YOUR_APP_ID"    # Replace with your Clarifai App ID
    model_id = "YOUR_MODEL_ID"  # Replace with your Clarifai Model ID
    
    response = mandelbulb_generation.upload_mandelbulb_image("modified_mandelbulb.png", user_id, pat_key, app_id, model_id)
    print("Upload response:", response)

if __name__ == '__main__':
    main()

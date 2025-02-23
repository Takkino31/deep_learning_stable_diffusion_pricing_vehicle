import torch
from diffusers import StableDiffusionPipeline

# Charger Stable Diffusion une seule fois pour éviter le rechargement à chaque appel
device = torch.device("cpu")
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)

def generate_prompt(vehicle_data):
    """Génère un prompt Stable Diffusion basé sur les caractéristiques du véhicule."""
    return (f"A {vehicle_data['paint_color']} {vehicle_data['year']} {vehicle_data['manufacturer']} "
            f"{vehicle_data['model']} in {vehicle_data['condition']} condition, "
            f"with {vehicle_data['cylinders']} cylinders and {vehicle_data['transmission']} transmission.")

def generate_car_image(vehicle_data):
    """Génère une image de voiture à partir des caractéristiques fournies."""
    prompt = generate_prompt(vehicle_data)
    image = pipe(prompt).images[0]
    return image


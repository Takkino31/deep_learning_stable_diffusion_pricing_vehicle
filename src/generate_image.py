import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
from predict_price import predict_car_price

def generate_prompt(vehicle_data):
    """Génère un prompt Stable Diffusion basé sur les caractéristiques du véhicule."""
    return (f"A {vehicle_data['paint_color']} {vehicle_data['year']} {vehicle_data['manufacturer']} "
            f"{vehicle_data['model']} in {vehicle_data['condition']} condition, "
            f"with {vehicle_data['cylinders']} cylinders and {vehicle_data['transmission']} transmission.")

# Informations sur la voiture avec des valeurs par défaut pour les champs manquants
vehicle_info = {
    "paint_color": "red",
    "year": "2022",
    "manufacturer": "Toyota",
    "model": "Corolla",
    "condition": "excellent",
    "cylinders": "4",
    "transmission": "automatic"
}
# Prédire le prix
predicted_price = predict_car_price(vehicle_info)
print(f"Prix estimé du véhicule : {predicted_price} CFA")



# Force PyTorch à utiliser le CPU
device = torch.device("cpu")
# Charger Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)



# Générer l'image
prompt = generate_prompt(vehicle_info)
image = pipe(prompt).images[0]

# Afficher l'image
plt.imshow(image)
plt.axis("off")
plt.show()
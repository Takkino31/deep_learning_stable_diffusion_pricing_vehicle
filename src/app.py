import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from predict_price import predict_car_price
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def generate_prompt(vehicle_data):
    """Génère un prompt Stable Diffusion basé sur les caractéristiques du véhicule."""
    return (f"A {vehicle_data['paint_color']} {vehicle_data['year']} {vehicle_data['manufacturer']} "
            f"{vehicle_data['model']} in {vehicle_data['condition']} condition, "
            f"with {vehicle_data['cylinders']} cylinders and {vehicle_data['transmission']} transmission.")

# Initialisation de l'interface Streamlit
st.title("Generateur de prix et d'images de voitures.")

# Formulaire pour entrer les caractéristiques du véhicule
with st.form("car_form"):
    paint_color = st.text_input("Couleur", "red")
    year = st.text_input("Année", "2022")
    manufacturer = st.text_input("Marque", "Toyota")
    model = st.text_input("Modèle", "Corolla")
    condition = st.selectbox("État", ["new", "excellent", "good", "fair", "salvage"])
    cylinders = st.text_input("Cylindres", "4")
    transmission = st.selectbox("Transmission", ["automatic", "manual"])
    submit = st.form_submit_button("Générer")

if submit:
    vehicle_info = {
        "paint_color": paint_color,
        "year": year,
        "manufacturer": manufacturer,
        "model": model,
        "condition": condition,
        "cylinders": cylinders,
        "transmission": transmission
    }
    
    with st.spinner("Génération en cours... Veuillez patienter."):
        # Prédire le prix
        predicted_price = predict_car_price(vehicle_info)
        
        # Charger Stable Diffusion sur CPU uniquement lors de la génération
        device = torch.device("cpu")
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
        
        # Générer l'image
        prompt = generate_prompt(vehicle_info)
        image = pipe(prompt).images[0]
        
        # Sauvegarder l'image en mémoire pour affichage
        img_bytes = BytesIO()
        image.save(img_bytes, format="PNG")
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode()
        
    # Affichage des résultats
    st.success(f"Prix estimé : {predicted_price}")
    st.image(image, caption="Image générée", use_column_width=True)

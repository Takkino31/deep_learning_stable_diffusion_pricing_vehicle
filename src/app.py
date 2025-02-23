import streamlit as st
import torch
import base64
import tempfile
from io import BytesIO
from diffusers import StableDiffusionPipeline
from predict_price import predict_car_price
from generate_image import generate_car_image

# Initialisation de l'interface Streamlit
st.title("Générateur de Prix et d'Images de Voitures")

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
    # Création du dictionnaire des caractéristiques du véhicule
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
        # Prédiction du prix
        predicted_price = predict_car_price(vehicle_info)

        # Génération de l'image
        image = generate_car_image(vehicle_info)

        # Sauvegarde de l'image pour affichage
        img_bytes = BytesIO()
        image.save(img_bytes, format="PNG")
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode()

    # Affichage des résultats
    st.success(f"Prix estimé : {predicted_price} CFA")
    st.image(image, caption="Image générée", use_container_width=True)

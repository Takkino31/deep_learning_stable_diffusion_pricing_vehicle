import tensorflow as tf
import numpy as np

tf.config.set_visible_devices([], 'GPU')

# Enregistrer la fonction de perte MSE avant de charger le modèle
from keras.saving import register_keras_serializable

@register_keras_serializable()
def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

# Charger le modèle en incluant la fonction personnalisée
model = tf.keras.models.load_model("models/voiture_pricing_model.h5", custom_objects={"mse": mse})

from sklearn.preprocessing import LabelEncoder

def preprocess_input(vehicle_data):
    """Transforme les données de la voiture en format utilisable par le modèle."""
    # Utilisation de LabelEncoder pour les variables catégorielles
    le = LabelEncoder()

    # Convertir les variables catégorielles en valeurs numériques
    vehicle_data["condition"] = le.fit_transform([vehicle_data["condition"]])[0]
    vehicle_data["transmission"] = le.fit_transform([vehicle_data["transmission"]])[0]
    vehicle_data["manufacturer"] = le.fit_transform([vehicle_data["manufacturer"]])[0]
    vehicle_data["model"] = le.fit_transform([vehicle_data["model"]])[0]
    vehicle_data["paint_color"] = le.fit_transform([vehicle_data["paint_color"]])[0]

    # Convertir 'year' et 'cylinders' en entiers
    vehicle_data["year"] = int(vehicle_data["year"])
    vehicle_data["cylinders"] = int(vehicle_data["cylinders"])

    feature_order = ["year", "cylinders", "condition", "transmission", "manufacturer", "model", "paint_color"]
    
    # Transformer les données en un tableau de valeurs numériques
    numerical_data = [
        vehicle_data["year"],
        vehicle_data["cylinders"],
        vehicle_data["condition"],
        vehicle_data["transmission"],
        vehicle_data["manufacturer"],
        vehicle_data["model"],
        vehicle_data["paint_color"]
    ]
    
    return np.array([numerical_data])


def predict_car_price(vehicle_data):
    """Prédit le prix d'une voiture à partir de ses caractéristiques."""
    input_data = preprocess_input(vehicle_data)
    predicted_price = model.predict(input_data)[0][0]
    return round(predicted_price, 2)
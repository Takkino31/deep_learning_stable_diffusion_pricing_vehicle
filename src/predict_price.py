import tensorflow as tf
import numpy as np
from keras.saving import register_keras_serializable
from sklearn.preprocessing import LabelEncoder

# Désactivation du GPU pour éviter les conflits
tf.config.set_visible_devices([], 'GPU')

@register_keras_serializable()
def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

# Charger le modèle de prédiction
model = tf.keras.models.load_model("models/voiture_pricing_model.h5", custom_objects={"mse": mse})

def preprocess_input(vehicle_data):
    """Transforme les données de la voiture en format utilisable par le modèle."""
    le = LabelEncoder()

    vehicle_data["condition"] = le.fit_transform([vehicle_data["condition"]])[0]
    vehicle_data["transmission"] = le.fit_transform([vehicle_data["transmission"]])[0]
    vehicle_data["manufacturer"] = le.fit_transform([vehicle_data["manufacturer"]])[0]
    vehicle_data["model"] = le.fit_transform([vehicle_data["model"]])[0]
    vehicle_data["paint_color"] = le.fit_transform([vehicle_data["paint_color"]])[0]

    vehicle_data["year"] = int(vehicle_data["year"])
    vehicle_data["cylinders"] = int(vehicle_data["cylinders"])

    feature_order = ["year", "cylinders", "condition", "transmission", "manufacturer", "model", "paint_color"]
    
    numerical_data = [vehicle_data[feature] for feature in feature_order]

    return np.array([numerical_data])

def predict_car_price(vehicle_data):
    """Prédit le prix d'une voiture en fonction de ses caractéristiques."""
    input_data = preprocess_input(vehicle_data)
    predicted_price = model.predict(input_data)[0][0]
    return round(predicted_price, 2)

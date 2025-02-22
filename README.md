Voici le fichier `README.md` en un seul bloc :

```md
# Car Pricing with Stable Diffusion

Ce projet permet de générer une image d'une voiture avec Stable Diffusion et de prédire son prix à l'aide d'un modèle de machine learning.

## Structure du projet

```
car_pricing_stable_diffusion/
│
├── env/                         # Dossier pour l'environnement virtuel
├── models/
│   └── voiture_pricing_model.h5  # Modèle de prédiction des prix des voitures
├── src/
│   ├── generate_image.py         # Script pour générer l'image et le prix
│   └── predict_price.py          # Script pour prédire le prix
├── README.md                     # Instructions du projet
├── requirements.txt              # Liste des bibliothèques nécessaires
└── .gitignore                    # Fichier pour ignorer certains fichiers (comme env)
```

## Installation
```

### 1. Cloner le projet

```bash
git clone https://github.com/Takkino31/deep_learning_stable_diffusion_pricing_vehicle
cd car_pricing_stable_diffusion
```

### 2. Créer un environnement virtuel

```
Pour Linux/macOS :
```

```bash
python3 -m venv env
source env/bin/activate
```

Pour Windows :

```bash
python -m venv env
env\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Ajouter le modèle

Placez le modèle de prédiction des prix `voiture_pricing_model.h5` dans le dossier `models/`.

## Utilisation

1. Modifiez les informations sur la voiture dans `src/generate_image.py` :

```python
vehicle_info = {
    "paint_color": "blue",
    "year": 2020,
    "manufacturer": "Ford",
    "model": "Mustang",
    "condition": "good",
    "cylinders": 8,
    "transmission": "manual"
}
```

2. Exécutez le script pour générer l'image et afficher le prix prédit :

```bash
python src/generate_image.py
```

## Exemples

Le script générera une image de la voiture décrite et affichera le prix prédit à l'aide du modèle `voiture_pricing_model.h5`.

## Dépendances

- diffusers
- transformers
- torch
- tensorflow
- Pillow
- matplotlib
```

Cela inclut toutes les informations nécessaires pour la configuration, l'utilisation et l'exécution du projet.
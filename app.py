import os
from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import base64
from io import BytesIO

app = Flask(__name__)

# Chargement du modèle TensorFlow
model = tf.keras.models.load_model('traffic_sign_model.h5')

# Définition des classes (les mêmes que celles utilisées pour l'entraînement)
classes = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing veh over 3.5 tons',
    11: 'Right-of-way at intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Veh > 3.5 tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve left',
    20: 'Dangerous curve right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End speed + passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End no passing veh > 3.5 tons'
}

def prepare_image(image):
    """
    Cette fonction redimensionne et prétraiter l'image pour qu'elle soit compatible avec le modèle.
    """
    image = image.resize((30, 30))  # Redimensionner l'image à 30x30 comme attendu par le modèle
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # Ajouter une dimension pour le batch
    return image

@app.route('/predict', methods=['POST'])
def predict():
    """
    Cette route reçoit une image en base64, prédit la classe et retourne la prédiction.
    """
    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400

    # Récupérer l'image de la requête en base64
    image_data = request.json['image']
    
    # Décoder l'image depuis la chaîne base64
    try:
        img_data = base64.b64decode(image_data)
        img = Image.open(BytesIO(img_data))  # Charger l'image depuis le buffer
    except Exception as e:
        return jsonify({'error': f'Error decoding image: {str(e)}'}), 400

    # Prétraiter l'image pour qu'elle soit compatible avec le modèle
    processed_image = prepare_image(img)

    # Effectuer la prédiction avec le modèle
    prediction = np.argmax(model.predict(processed_image), axis=-1)

    # Récupérer le nom de la classe prédite
    predicted_class = classes[int(prediction)]
    
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    # Démarrer le serveur Flask sur le port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)

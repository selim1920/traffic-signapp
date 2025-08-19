# 🚦 Traffic Sign Recognition App

Traffic Sign Recognition App est une API Flask qui utilise un modèle TensorFlow pour détecter et classifier des panneaux de signalisation à partir d'images encodées en Base64.

---

## 🛠️ Technologies utilisées

- **Backend** : Python, Flask  
- **Machine Learning** : TensorFlow / Keras  
- **Manipulation d'images** : Pillow, NumPy  
- **Déploiement** : Docker, Kubernetes (fichiers YAML fournis)  

---

## 📂 Structure du projet

```text
traffic-signapp/
│
├── app.py                  # API Flask pour les prédictions
├── traffic_sign_model.h5   # Modèle pré-entraîné TensorFlow
├── requirements.txt        # Dépendances Python
├── Dockerfile              # Pour containeriser l'application
├── index.html              # Exemple d'interface web
├── k8s-yaml/               # Fichiers YAML pour Kubernetes
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── deployment-model.yaml
│   ├── service-model.yaml
│   ├── configmap.yaml
│   └── configmap-model.yaml
└── README.md
⚡ Installation et lancement
Prérequis
Python 3.9+

Docker (optionnel pour container)

Kubernetes (optionnel pour cluster)

Installation locale
bash
Copy
Edit
git clone https://github.com/selim1920/traffic-signapp.git
cd traffic-signapp
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
python app.py
L'API sera disponible sur http://localhost:5000.

🚀 Endpoints de l'API
1. Accueil
sql
Copy
Edit
GET /
Réponse :

json
Copy
Edit
{
  "message": "Welcome to our Traffic Sign Recognition API!"
}
2. Prédiction
bash
Copy
Edit
POST /predict
Paramètres
image : Image encodée en Base64

Exemple avec curl
bash
Copy
Edit
curl -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d '{"image": "<base64_string_here>"}'
Réponse
json
Copy
Edit
{
  "prediction": "Speed limit (50km/h)"
}
🖼️ Prétraitement des images
Redimensionnement à 30x30 pixels

Conversion en tableau NumPy

Ajout d'une dimension pour le batch

📝 Classes de panneaux reconnues
L'API prend en charge 43 classes, par exemple :

Speed limit (20km/h)

Stop

Yield

No passing

Pedestrians

Bicycles crossing

Roundabout mandatory

(Liste complète dans app.py)

🐳 Déploiement avec Docker
bash
Copy
Edit
docker build -t traffic-signapp .
docker run -p 5000:5000 traffic-signapp
☸️ Déploiement avec Kubernetes
Fichiers YAML disponibles dans k8s-yaml/.

bash
Copy
Edit
kubectl apply -f k8s-yaml/configmap.yaml
kubectl apply -f k8s-yaml/deployment.yaml
kubectl apply -f k8s-yaml/service.yaml
🔗 Liens utiles
Documentation Flask

TensorFlow Keras

Pillow (PIL)


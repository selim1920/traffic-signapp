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

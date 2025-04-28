# Image légère Python
FROM python:3.8-slim

# Créer un répertoire de travail
WORKDIR /app

# Copier requirements.txt et installer
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copier tout le reste de l'application
COPY . .

# Exposer le port pour Flask
EXPOSE 5000

# Lancer l'application
CMD ["python", "app.py"]

# Utiliser l'image officielle de Python comme image de base
FROM python:3.9-slim

# Définir le répertoire de travail à l'intérieur du conteneur
WORKDIR /app

# Copier le fichier requirements.txt dans le répertoire de travail
COPY requirements.txt .

# Installer les dépendances listées dans requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copier le script Python de nettoyage des données dans le conteneur
COPY clean_data.py .

# Définir la commande d'exécution par défaut (exécuter le script Python)
CMD ["python", "clean_data.py"]

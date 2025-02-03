# Utiliser l'image officielle de Jupyter
FROM jupyter/base-notebook:latest

# Définir le répertoire de travail dans le conteneur
WORKDIR /work

# Copier le notebook et les données dans le conteneur
COPY /work/clean_data.ipynb /work/
COPY /work/datasets /work/datasets

# Copier le fichier requirements.txt dans le répertoire de travail
COPY requirements.txt /tmp/

# Vérifier que les fichiers sont bien copiés
RUN ls -l /work/

# Installer les dépendances
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Convertir le notebook en script Python
RUN jupyter nbconvert --to script /work/clean_data.ipynb

# Exécuter le script Python
CMD ["python", "/work/clean_data.py"]

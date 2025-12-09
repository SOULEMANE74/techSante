FROM python:3.11-slim

# 2. Dossier de travail
WORKDIR /app

# 3. Installation des outils systèmes 

RUN apt-get update && apt-get install -y \
    build-essential \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copie et Installation des dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copie du code
COPY . .

# 6. Initialisation de la DB 
RUN python init_db.py

# 7. GESTION DES DROITS 

RUN chmod -R 777 .

# 8. Création de l'utilisateur 
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 9. Lancement sur le port 7860
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
# Super 

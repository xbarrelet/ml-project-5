# Projet 5: Catégorisez automatiquement des questions
Ce répositoire contient les fichiers nécessaires pour le projet 5 de la formation Machine Learning Engineer d'OpenClassrooms.  
Il contient les dossiers suivants:
- .github/workflows: le fichier yaml pour le workflow de CI/CD.
- inferring_api: La webapp Flask qui permet d'inférer les catégories des questions posées.
- notebooks: Les notebooks provenant des scripts du dossier racine.


Fais un proper README.md pour ce projet.


- Add variable SITE_API_KEY in .env file if more than 2500 results fetched desired.
- Start local MLFlow server: 
mlflow server --host 127.0.0.1 --port 8080

MLOPS: https://ashutoshtripathi.com/2021/08/18/mlops-a-complete-guide-to-machine-learning-operations-mlops-vs-devops/
# Projet 5: Catégorisez automatiquement des questions
Ce répositoire contient les fichiers nécessaires pour le projet 5 de la formation Machine Learning Engineer d'OpenClassrooms.  
Il se trouve à l'addresse suivante: https://github.com/xbarrelet/ml-project-5.  
Plus d'information à son sujet résident sur la page: https://openclassrooms.com/fr/projects/1510.

## Structure
Il contient les dossiers suivants:
- .github/workflows: le fichier yaml pour le workflow de CI/CD.
- inferring_api: La webapp Flask qui permet d'inférer les catégories des questions posées + sa configuration de container pour son déployement sur AWS + son interface cliente locale
- scripts: Les scripts originaux utilisés pendant la majeure partie du développement.

Les notebooks et documents pour ce projet se trouvent dans le dossier racine. Les questions pour les notebooks se trouvent déjà dans le dossier racine et répositoire git pour gagner du temps.

## Divers
- Pour récupérer plus de 2500 questions de Stackoverflow via requête API il va falloir d'abord créer une clé API sur ce site puis l'ajouter dans le fichier .env en tant que variable SITE_API_KEY. 
- Pour démarrer un serveur MLFlow local après avoir installé les dépendances du fichier requirements.txt il suffit de lancer la commande suivante dans l'environnement virtuel:
```bash
mlflow server --host 127.0.0.1 --port 8080

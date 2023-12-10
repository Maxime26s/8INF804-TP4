# 8INF804 - TP4

## Auteurs

- Maya Legris (LEGM15600100)
- Tifenn Le Gourriérec (LEGT08590100)
- Maxime Simard (SIMM26050001)

## Prérequis

Python >= 3.9

## Installation

### GitHub

- Cloner le repository du projet

### Moodle

- Décompresser le fichier zip du projet

## Utilisation

- Toutes les commandes sont à exécuter à partir de la racine du projet
- Pour utiliser le projet, vous devez vous-même télécharger le dataset, et répéter les étapes suivantes:
  - Garder seulement le dossier d'image appelé "NORMAL"
  - Faire la strcture de fichier suivante: `./images/NORMAL/*.jpg`

### Environnement virtuel

- Afin de facilité l'utilisation du projet, il est recommandé d'utiliser un environnement virtuel
- Pour se faire, exécuter la commande suivante:

  `python3 -m venv venv`

- Activer l'environnement virtuel sur Windows:

  `.\venv\Scripts\activate`

- Pour installer les dépendences, exécuter la commande suivante:

  `pip install -r requirements.txt`

### Arguments

    usage: tp4 [-h] [--n_epochs N_EPOCHS] [--batch_size BATCH_SIZE] [--lr LR] [--b1 B1] [--b2 B2] [--latent_dim LATENT_DIM] [--img_size IMG_SIZE] [--channels CHANNELS] [--normalization_choice NORMALIZATION_CHOICE] [--load_model LOAD_MODEL] [--model_path MODEL_PATH] [--mode MODE] [--current_epoch CURRENT_EPOCH]

    tp4

    options:
    -h, --help                                      show this help message and exit
    --n_epochs N_EPOCHS                             Number of epochs
    --batch_size BATCH_SIZE                         Batch size
    --lr LR                                         Learning Rate
    --b1 B1                                         Beta 1 hyperparameter for Adam optimizers
    --b2 B2                                         Beta 2 hyperparameter for Adam optimizers
    --latent_dim LATENT_DIM                         Latent dimension
    --img_size IMG_SIZE                             Image size (width/height)
    --channels CHANNELS                             Number of image channels
    --normalization_choice NORMALIZATION_CHOICE     Normalization choice (0-5)
    --load_model LOAD_MODEL                         Load saved model (0: No, 1: Yes)
    --model_path MODEL_PATH                         Path to model folder
    --mode MODE                                     Mode: train or eval
    --current_epoch CURRENT_EPOCH                   Current epoch (for loading saved models)

### Exemple d'utilisation

- Les images d'entraînement doivent être placées dans le dossier ./images/. Elles peuvent être au format .png, .jpg ou .jpeg.

- Les modèles entraînés sont sauvegardés par défaut dans le dossier ./models.

- Pour exécuter le programme en mode d'entraînement :

  `python3 tp4 --mode train`

- Pour évaluer et générer des images à l'aide d'un modèle pré-entraîné :

  `python3 tp4 --mode eval --load_model 1 --model_path ./models --current_epoch 4000`

- Pour personnaliser davantage la configuration de l'entraînement, utilisez les arguments disponibles, par exemple :

  `python3 tp4 --mode train --n_epochs 10000 --batch_size 2048 --lr 0.0002 --latent_dim 64 --img_size 64 --channels 1 --normalization_choice 4`

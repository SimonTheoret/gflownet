Lien vers le repo github: https://github.com/SimonTheoret/gflownet

# Installation
## Installation avec Pip
Pour installer GFlowChess avec pip, il suffit de faire les commande suivante:
```
pip install .
pip install chess
``` 
Une étape supplémentaire est requise pour entraîner le modèle (voir
section Stockfish)

## Installation avec Pipenv
Pour installer GFlowChess avec pipenv, il suffit de faire les commande suivante:
```
pip install .
pipenv install
``` 
Une étape supplémentaire est requise pour entraîner le modèle (voir
section Stockfish)

## Dépendances
Les dépendances nécessaire au bon fonctionnement de GFlowChess sont
contenues dans le fichier Pipfile. Une unique dépendance
supplémentaire est ajoutée aux dépendances initiale du package
GFlowNet: python chess: https://pypi.org/project/chess/

## Stockfish
Pour pouvoir entraîner le modèle, il est nécessaire d'avoir défini la
variable d'environnement `STOCKFISH`, qui dénote le chemin vers le
fichier binaire de Stockfish.

Voici le lien pour télécharger le fichier binaire de Stockfish:
https://stockfishchess.org/download/

## Versions Python et CUDA
Il est recommandé d'utilise la version python 3.10 ainsi que cuda 11.8

# Entraînement
Pour entraîner le modèle, il suffit d'utiliser les scripts `run_x.sh`,
où `x` peut être `fl` pour Forward Looking, `fm` pour Flow Matching et
`tb` pour Trajectory Balance.


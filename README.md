# Automatic Candy Selector (ACS)

Projet de vision par ordinateur — **IN450, ESISAR (2026–2027)**.

L'ACS détecte les bonbons dans une photographie, puis reconnaît leur **type** et leur **variante de couleur**. Nous utilisons trois méthodes : **YOLOv8n** pour trouver les bonbons, **ConvNeXt-Tiny** pour identifier le type et **HSV + KNN** pour choisir la variante. Une interface **Java Swing** communique avec les modèles Python.

Le catalogue contient **6 types** et **21 combinaisons type–variante**. Pour les oursons, il existe une seule variante `rouge`.

## Documents et code

- [Classification : préparation, entraînement et évaluation](docs/classification/README.md)
- [Détection : préparation, entraînement et évaluation](docs/detection/README.md)
- Code de classification : `src/classification/` et `src/color/`.
- Code de détection : `src/detection/`.
- Configurations : `configs/` ; évaluations : `src/evaluation/`.
- Interface : `java-app/src/main/java/MainFrame.java` ; service Python : `worker.py`.

## Fonctionnement

```text
Image / caméra
      |
      v
YOLOv8n (boîtes des bonbons)
      |
      v
Recadrage de chaque boîte
      |
      v
ConvNeXt-Tiny (6 types)
      |
      v
HSV + KNN (variante selon le type)
      |
      v
Image annotée et résultats dans l'interface Java
```

Les deux classifications sont évaluées sur des **photos de bonbons isolés**. Pour la détection, les métriques sont calculées sur des **scènes contenant plusieurs bonbons**. Le rapport IN450 présente ces résultats séparément ; aucune mesure chiffrée de la chaîne complète sur des scènes n'est fournie.

## Installation

Depuis la **racine du dépôt** :

```bash
# Python 3.11 recommandé ; environnement virtuel ou Conda possible
python -m pip install -r requirements.txt
```

PyTorch et `torchvision` doivent correspondre au système et, si nécessaire, à la version CUDA. Une connexion Internet peut être nécessaire pour installer les bibliothèques ou télécharger les poids pré-entraînés lors d'un nouvel entraînement.

**Important :** les images des datasets, les fichiers de modèles (`.pth`, `.pt`, `.joblib`) et les résultats sont ignorés par Git. Un simple clonage du dépôt ne suffit donc pas à reproduire les évaluations. Il faut également disposer des dossiers `datasets/v2/` et `models/`, ou les recevoir dans l'archive de remise.

## Vérification rapide

```bash
python -m src.verify --strict
python -m src.classification.train --check-only
python -m src.color.train --check-only
python -m src.detection.train --check-only
```

La première commande vérifie les fichiers nécessaires ; les autres contrôlent les datasets sans lancer d'entraînement.

## Reproduire les évaluations (modèles déjà entraînés)

Exécuter les commandes depuis la racine du projet :

```bash
# 1) Classification du type (ConvNeXt)
python -m src.evaluation.classification --split test

# 2) Classification de la couleur avec le type réel (HSV/KNN)
python -m src.evaluation.color --split test

# 3) Type puis couleur : 21 combinaisons possibles
python -m src.evaluation.joint_classification --split test

# 4) Détection seule (YOLOv8n)
python -m src.detection.evaluate --split test
```

Pour exécuter les mêmes évaluations sur les données de validation, remplacer `--split test` par `--split val`.

Les rapports (métriques, matrices de confusion et images) sont créés dans `results/evaluation/`. **Ces commandes ne réentraînent pas les modèles.** Elles nécessitent les checkpoints suivants :

```text
models/classification/convnext_tiny_best.pth
models/color/color_classifier.joblib
models/detection/yolo_best.pt
```

## Entraînement (optionnel)

Ne pas lancer ces commandes si l'objectif est seulement de vérifier les résultats du rapport : un nouveau modèle peut remplacer le checkpoint actuel.

```bash
python -m src.classification.train
python -m src.color.train
python -m src.detection.train --device cpu
```

Les paramètres se trouvent dans `configs/classification.yaml`, `configs/color.yaml` et `configs/detection.yaml`. Les méthodes et les dossiers attendus sont détaillés dans les deux README spécialisés.

## Essayer l'application

Une fois les trois modèles disponibles, on peut traiter une photographie :

```bash
python -m src.detection.integrated_pipeline --source chemin/vers/photo.jpg
```

Pour utiliser l'interface Java, un JDK avec `java` et `javac` est nécessaire :

```bash
export ACS_PROJECT_ROOT="$(pwd)"
export ACS_PYTHON="$(command -v python)"
mkdir -p java-app/build
javac -encoding UTF-8 -d java-app/build java-app/src/main/java/MainFrame.java
java -cp java-app/build MainFrame
```

L'interface permet de choisir une image, de la déposer dans la fenêtre ou d'utiliser une caméra si elle est disponible. Les modèles restent chargés pendant la session.

## Organisation simplifiée

```text
AutomaticCandySelector/
├── README.md
├── docs/
│   ├── classification/README.md
│   └── detection/README.md
├── configs/
├── datasets/v2/classification/processed/
├── datasets/v2/detection/yolo/
├── models/
├── results/
├── src/
│   ├── classification/
│   ├── color/
│   ├── detection/
│   ├── evaluation/
│   └── data/
├── java-app/
└── worker.py
```

**Dépôt distant :** https://github.com/EsisarProjects/AutomaticCandySelector

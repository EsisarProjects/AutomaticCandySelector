# Partie 1 — Classification des bonbons

Ce document explique comment reproduire la **classification** du projet ACS. Les commandes doivent être exécutées depuis la **racine du dépôt** (`AutomaticCandySelector/`).

## Méthode

Nous reconnaissons d'abord le **type du bonbon** avec **ConvNeXt-Tiny**, pré-entraîné puis adapté à nos **6 types**. Ensuite, **HSV + KNN (`k=3`)** détermine la variante de couleur en fonction du type. Il existe **21 combinaisons type–variante** dans `configs/classes.yaml`.

Pour `fraise` et `oeuf`, une seule variante est possible : le catalogue donne directement cette variante. Pour `schtroumpf`, la variante correspond à la couleur du bonnet. Le classificateur de couleur utilise la segmentation HSV et des caractéristiques de couleur ; certaines régions claires ou très petites sont difficiles à analyser.

Les paramètres utilisés sont dans `configs/classification.yaml` et `configs/color.yaml` : images de **224 × 224**, ConvNeXt-Tiny, **AdamW**, taux `0.0002`, jusqu'à **10 époques**, `batch_size=32`, et HSV/KNN avec **3 voisins**.

## Organisation des images

La classification attend une arborescence de ce type :

```text
datasets/v2/classification/processed/
├── crocodile/
│   ├── bleu/
│   │   ├── train/*.jpg
│   │   ├── val/*.jpg
│   │   └── test/*.jpg
│   └── ...
├── dragibus/
├── fraise/
├── oeuf/
├── ourson/       # blanc, jaune, orange, rouge, vert
└── schtroumpf/
```

Les mêmes trois dossiers (`train`, `val`, `test`) sont attendus **pour chaque variante**. Le test est séparé et ne doit pas être utilisé pour choisir les paramètres du modèle.

Si les images sont encore dans un dossier externe `Classification/Train_Val` et `Classification/Test`, on peut les importer (copie et séparation de `Train_Val` en `train`/`val`) :

```bash
python -m src.data.import_classification --source /chemin/vers/Classification
# Après vérification du récapitulatif :
python -m src.data.import_classification --source /chemin/vers/Classification --apply
```

Ne pas refaire l'importation si le dossier `processed/` contient déjà les images finales. La séparation automatique ne garantit pas l'indépendance de photographies prises pendant une même séance.

## Préparer l'environnement

```bash
python -m pip install -r requirements.txt
python -m src.classification.train --check-only
python -m src.color.train --check-only
```

Ces deux dernières commandes vérifient les images **sans lancer l'entraînement**.

## A. Reproduire les résultats avec les modèles existants (recommandé)

Les fichiers nécessaires sont :

```text
models/classification/convnext_tiny_best.pth
models/color/color_classifier.joblib
```

### 1. Type : ConvNeXt (6 classes)

```bash
python -m src.evaluation.classification --split test
```

Résultats : `results/evaluation/classification/test/` (métriques et matrice de confusion).

### 2. Couleur : HSV/KNN avec le type réel

```bash
python -m src.evaluation.color --split test
```

Résultats : `results/evaluation/color/test/`. **Attention :** cette évaluation fournit le type réel au modèle de couleur. Elle ne mesure donc pas les erreurs du ConvNeXt.

### 3. Classification conjointe : type + variante

```bash
python -m src.evaluation.joint_classification --split test
```

Résultats : `results/evaluation/joint_classification/test/`. Cette fois, le HSV/KNN reçoit le **type prédit par ConvNeXt**, comme pendant l'utilisation normale du classificateur.

Pour consulter les performances sur la validation, remplacer `test` par `val` dans ces trois commandes. **Aucune des trois commandes ne modifie les poids.**

## B. Entraîner les modèles (optionnel)

Ces commandes servent à produire de nouveaux checkpoints. Elles ne sont pas nécessaires si l'on veut seulement reproduire le rapport.

```bash
# Apprentissage des 6 types ; enregistre le meilleur ConvNeXt
python -m src.classification.train

# Apprentissage des variantes avec HSV/KNN
python -m src.color.train
```

Le ConvNeXt utilise `train` et choisit son checkpoint avec `val`. Le modèle HSV/KNN est ajusté avec `train` et contrôlé sur `val`. Les checkpoints sont enregistrés sous `models/` ; les courbes et journaux sous `results/training/`.

Pour vérifier uniquement les checkpoints déjà entraînés via les commandes propres aux entraîneurs :

```bash
python -m src.classification.train --test-only
python -m src.color.train --test-only
```

Ces commandes produisent un résumé de test plus simple que les évaluateurs de la partie A. Pour les métriques détaillées (Macro-F1, matrices), utiliser **la partie A**.

## C. Validation croisée (expérience facultative)

Nous avons réalisé une validation croisée de **3 folds**, avec **train + val uniquement**. Les fichiers strictement identiques sont regroupés ; différentes photos d'un même bonbon peuvent cependant rester dans plusieurs folds.

```bash
# Vérifier les folds sans entraîner
python -m src.evaluation.cross_validation --check-only

# Réexécuter les trois entraînements (facultatif et coûteux)
python -m src.evaluation.cross_validation --folds 3
```

Résumé : `results/evaluation/classification/cross_validation/summary.json`. Le jeu `test` n'est jamais utilisé pendant cette procédure. Cette étape n'est **pas nécessaire** pour lancer les évaluations des checkpoints existants.

## Interprétation des résultats du rapport

Dans nos essais, ConvNeXt a obtenu **77,88 %** d'accuracy sur le test (6 types), HSV/KNN **82,42 %** avec le type réel, et la classification complète **63,79 %**. Ces scores proviennent d'images individuelles, sans recadrage automatique du YOLO. Ils peuvent donc être différents pendant l'utilisation de la chaîne complète.

[Retour au README principal](../../README.md)

# Partie 2 — Détection des bonbons

Ce document explique comment reproduire la **détection YOLOv8n** du projet ACS. Exécuter les commandes depuis la **racine du dépôt** (`AutomaticCandySelector/`).

## Méthode

Nous utilisons **YOLOv8n** avec des poids pré-entraînés pour trouver les bonbons dans une image. Le détecteur reconnaît **une seule classe : `bonbon`**. Il retourne une boîte autour de chaque bonbon ; le type et la variante sont reconnus ensuite par les composants de classification.

Les images ont été annotées dans **Roboflow** puis exportées au **format YOLOv8**. Le dataset utilisé dans le rapport contient **837 images de train, 80 de validation et 40 de test**. Une image peut contenir plusieurs bonbons.

Les paramètres sont centralisés dans `configs/detection.yaml` : **YOLOv8n**, taille **640 × 640**, jusqu'à **30 époques**, `batch_size=2`, **AdamW**, taux initial `0.001`, et apprentissage sur **CPU** dans l'expérience présentée. Les augmentations supplémentaires sont désactivées dans le script Python (`augment: false`).

## Organisation des images et annotations

```text
datasets/v2/detection/yolo/
├── data.yaml
├── train/
│   ├── images/*.jpg
│   └── labels/*.txt
├── val/
│   ├── images/*.jpg
│   └── labels/*.txt
└── test/
    ├── images/*.jpg
    └── labels/*.txt
```

Chaque fichier `.txt` contient les boîtes au format YOLO (`classe x_centre y_centre largeur hauteur`, coordonnées normalisées). La classe a l'indice `0`.

Exemple minimal de `data.yaml` :

```yaml
train: train/images
val: val/images
test: test/images
nc: 1
names:
  0: bonbon
```

**Important :** Roboflow peut exporter le dossier sous le nom `valid` ; le renommer en `val` et corriger `data.yaml`. Si le nom de classe exporté est `bonbom`, le corriger en **`bonbon`** dans `data.yaml` : c'est le nom vérifié par le code ACS. Il n'est pas nécessaire de changer les indices `0` dans les labels.

## Préparer l'environnement

```bash
python -m pip install -r requirements.txt
python -m src.detection.train --check-only
```

La deuxième commande vérifie les chemins, les splits et les annotations **sans entraîner**.

## A. Reproduire l'évaluation avec le modèle existant (recommandé)

Le checkpoint est attendu ici :

```text
models/detection/yolo_best.pt
```

Pour évaluer le modèle sur le jeu de test :

```bash
python -m src.detection.evaluate --split test
```

Pour la validation :

```bash
python -m src.detection.evaluate --split val
```

Les métriques et les figures sont enregistrées dans :

```text
results/evaluation/detection/test/
results/evaluation/detection/val/
```

Les métriques principales sont **Precision**, **Recall**, **mAP@50** et **mAP@50:95**. Les commandes chargent le checkpoint existant et **n'entraînent pas le modèle**.

Dans notre rapport, le test a donné **Precision 95,55 %**, **Recall 91,60 %**, **mAP@50 96,14 %** et **mAP@50:95 84,82 %**. Le nombre de photographies de test est limité (40), ce qui demande de la prudence dans l'interprétation.

## B. Entraîner le YOLO (optionnel)

Ne pas exécuter si l'objectif est seulement de vérifier les résultats du rapport, car un nouveau checkpoint peut remplacer le modèle actuel.

```bash
python -m src.detection.train --device cpu
```

Cette commande utilise les paramètres de `configs/detection.yaml` et enregistre le meilleur modèle dans `models/detection/yolo_best.pt`. Les courbes et les journaux se trouvent dans `results/training/detection/`.

Un test rapide de fonctionnement de l'entraîneur est également disponible :

```bash
python -m src.detection.train --smoke-test --device cpu
```

Ce test rapide utilise une seule époque et ne remplace pas le checkpoint principal.

## C. Exemple d'utilisation

Si les **trois checkpoints** YOLO, ConvNeXt et HSV/KNN sont disponibles, on peut essayer le traitement d'une photographie :

```bash
python -m src.detection.integrated_pipeline --source chemin/vers/photo.jpg
```

Cette commande sert à **visualiser des prédictions** : elle ne produit pas de mesure quantitative de la chaîne complète. Le résultat dépend notamment de la qualité des boîtes de détection. Un recadrage trop petit ou incomplet peut enlever des détails utiles pour la classification.

[Retour au README principal](../../README.md)

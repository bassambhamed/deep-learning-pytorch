# Module 1 — Fondations PyTorch, MLP & CNN

**Formation avancée en Deep Learning avec PyTorch — Computer Vision & Séries temporelles**
*Bassem Ben Hamed — ENETCOM, Université de Sfax (`bassem.benhamed@enetcom.usf.tn`)*

Travaux pratiques d'introduction : prise en main de PyTorch, puis construction et
évaluation d'un MLP et d'un CNN sur des **images médicales**. Les TP se suivent
dans l'ordre — chacun réutilise des acquis du précédent.

## Notebooks

| # | Notebook | Thèmes | Dataset |
|---|----------|--------|---------|
| 01 | `01_premiers_pas_pytorch.ipynb` | Tenseurs, broadcasting, interop NumPy, CPU/GPU, `autograd`, `nn.Module`, `Dataset`/`DataLoader`, boucle d'entraînement, `state_dict` | Régression synthétique (généré) |
| 02 | `02_mlp_images.ipynb` | Neurone & activations, forward/backprop, optimiseurs (SGD, momentum, Adam), régularisation (Dropout, BatchNorm, weight decay), initialisation (Xavier/Kaiming), métriques sur dataset déséquilibré | **PneumoniaMNIST** (radiographies thoraciques, binaire) |
| 03 | `03_cnn_vision_dermamnist.ipynb` | Convolution 2D, stride/padding, pooling, champ réceptif, BatchNorm, **data augmentation**, CNN VGG-like *from scratch*, viz filtres & feature maps, étude d'ablation | **DermaMNIST** (lésions cutanées RGB, 7 classes) |

## Datasets

Tous les jeux de données sont **téléchargés automatiquement** à la première exécution
(librairie [`medmnist`](https://medmnist.com/)) et mis en cache dans `./data/` :

- `pneumoniamnist.npz` — PneumoniaMNIST (TP 02), niveaux de gris 28×28, 2 classes.
- `dermamnist_128.npz` — DermaMNIST (TP 03), RGB 128×128, 7 classes (dérivé de HAM10000).
  Le TP 03 demande `size=128` ; sur une version ancienne de `medmnist` il retombe
  automatiquement sur les images 28×28 d'origine.

DermaMNIST est **fortement déséquilibré** (la classe *melanocytic nevi* domine) :
les TP suivent donc la **précision/rappel/F1 par classe** et la **macro-F1**, pas la
seule accuracy.

## Checkpoints produits

Sauvegardés dans `./checkpoints/` lors de l'exécution :

| Fichier | Produit par | Contenu |
|---------|-------------|---------|
| `tinymlp_weights.pt` | TP 01 | poids seuls (`state_dict`) |
| `tinymlp_full.pt` | TP 01 | poids + optimiseur + métadonnées (reprise d'entraînement) |
| `mlp_pneumonia_best.pt` | TP 02 | meilleur MLP sur PneumoniaMNIST |
| `smallvgg_dermamnist_best.pt` | TP 03 | meilleur CNN `SmallVGG` sur DermaMNIST |

## Pré-requis & installation

Python 3.10+ et les dépendances :

```bash
pip install torch torchvision torchmetrics medmnist matplotlib numpy
```

Un GPU CUDA accélère les TP 02 et 03 mais n'est pas obligatoire (le code bascule
automatiquement sur CPU).

## Exécution

Ouvrez chaque notebook dans Jupyter et exécutez les cellules dans l'ordre :

```bash
jupyter lab          # ou jupyter notebook
```

Suivez l'ordre 01 → 02 → 03 : le TP 03 (CNN/DermaMNIST) est aussi le point de départ
du **Module 2** (transfer learning sur le même dataset).

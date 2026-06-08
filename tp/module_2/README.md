# Module 2 — Transfer Learning, Séries temporelles & Explicabilité

**Formation avancée en Deep Learning avec PyTorch — Computer Vision & Séries temporelles**
*Bassem Ben Hamed — ENETCOM, Université de Sfax (`bassem.benhamed@enetcom.usf.tn`)*

Trois travaux pratiques : adapter un modèle pré-entraîné ImageNet à un domaine
médical (transfer learning), modéliser des **séries temporelles** avec des réseaux
récurrents, puis **expliquer** les prédictions d'un modèle vision (XAI).

## Notebooks

| # | Notebook | Thèmes | Données |
|---|----------|--------|---------|
| 04 | `04_transfer_learning_finetuning.ipynb` | ResNet18 ImageNet, feature extraction vs fine-tuning, LR différentiel, **OneCycleLR**, comparaison de 3 stratégies, catastrophic forgetting | **DermaMNIST** (réutilisé du TP 03) |
| 05 | `05_rnn_lstm_gru_series_temporelles.ipynb` | Équations RNN/LSTM/GRU, vanishing gradient, séquences de longueur variable, gradient clipping, BiLSTM | **UCI HAR**, **ETTh1**, **AI4I 2020** |
| 06 | `06_xai_explicabilite_grad_cam_shap.ipynb` | Saliency, **Grad-CAM**, Integrated Gradients, Occlusion, GradientSHAP, **LIME**, validation croisée avec **Captum** | **DermaMNIST** + modèle du TP 04 |

## Fil conducteur médical (TP 04 → 06)

Les TP 04 et 06 forment une chaîne sur **DermaMNIST** (lésions cutanées RGB, 7 classes) :

- Le **TP 04** réutilise le dataset du **TP 03** (Module 1) pour comparer un CNN
  entraîné *from scratch* à un ResNet18 pré-entraîné. Le domaine dermatoscopique
  étant **éloigné d'ImageNet**, c'est un cas où le fine-tuning profond se démarque
  nettement de la simple feature extraction.
- Le **TP 06** **audite** le modèle ResNet18 du TP 04 : vérifie qu'il « regarde la
  lésion » et non un artefact (marqueur, règle, coin d'image). Si le checkpoint du
  TP 04 est absent, un entraînement rapide est lancé automatiquement pour rendre le
  TP autonome.

## Datasets

### Vision (TP 04 & 06)
**DermaMNIST** via `medmnist`, téléchargé automatiquement. Réutilise de préférence
le cache du Module 1 (`../module_1/data/`), sinon un dossier `./data/` local.

### Séries temporelles (TP 05)
Téléchargés automatiquement dans `./data/` à la première exécution :

| Tâche | Dataset | Source | Dossier/fichier |
|-------|---------|--------|-----------------|
| Classification d'activité (accéléromètre 3 axes) | UCI HAR | archive.ics.uci.edu | `data/UCI_HAR/` |
| Forecasting multi-step (température transformateur) | ETTh1 | github.com/zhouhaoyi/ETDataset | `data/ETTh1.csv` |
| Détection d'anomalies (maintenance prédictive) | AI4I 2020 | archive.ics.uci.edu | `data/AI4I2020/` |

## Checkpoints produits

Sauvegardés dans `./checkpoints/` :

| Fichier | Produit par | Contenu |
|---------|-------------|---------|
| `resnet18_dermamnist_best.pt` | TP 04 | meilleure stratégie de transfer learning (poids + métadonnées). **Réutilisé par le TP 06.** |

## Pré-requis & installation

- Module 1 terminé (PyTorch, MLP, CNN).
- Dépendances :

```bash
pip install torch torchvision medmnist numpy pandas matplotlib scikit-learn
# optionnel pour le TP 06 (détecté automatiquement) :
pip install scikit-image captum shap
```

Un GPU CUDA accélère nettement les TP 04 et 06 (sinon bascule automatique sur CPU).

## Exécution

```bash
jupyter lab          # ou jupyter notebook
```

Ordre recommandé : **04 → 06** (chaîne vision DermaMNIST), **05** indépendant
(séries temporelles). Le TP 06 nécessite d'avoir exécuté le TP 04 au moins une fois
(ou laisse l'entraînement de secours s'exécuter).

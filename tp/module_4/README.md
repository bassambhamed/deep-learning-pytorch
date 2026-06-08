# Module 4 — GAN, Vision Transformers, Déploiement & Projet final

**Formation avancée en Deep Learning avec PyTorch — Computer Vision & Séries temporelles**
*Bassem Ben Hamed — ENETCOM, Université de Sfax (`bassem.benhamed@enetcom.usf.tn`)*

> 🚧 **En cours de développement.** Ce dossier est pour l'instant vide ; ce README décrit le
> contenu prévu, conforme à `plan.md` (Module 4) et aux conventions du dépôt. Les notebooks
> seront ajoutés au fil du développement.

Module final : les **GAN** et leurs variantes, les **Vision Transformers**, la **mise en
production** (TorchScript / ONNX), puis le **projet fil rouge** intégrant l'ensemble des
compétences de la formation.

## Notebooks prévus

La numérotation continue celle des modules précédents (01–09).

| # | Notebook (prévu) | Thèmes | Données envisagées |
|---|------------------|--------|--------------------|
| 10 | `10_gan_dcgan_cgan_pix2pix_srgan.ipynb` | Boucle d'entraînement adversariale (G vs D), loss (`BCEWithLogitsLoss`, Wasserstein/WGAN-GP), **DCGAN**, **cGAN** (`nn.Embedding` des labels), **Pix2Pix** (U-Net + PatchGAN, L1 + adversariale), **SRGAN** (perceptual loss VGG) | MNIST/CelebA, super-résolution médicale |
| 11 | `11_vision_transformers_vit_swin_segformer.ipynb` | **Patch embedding** (`einops`/`nn.Unfold`), positional encoding, `nn.MultiheadAttention`, **ViT**, **Swin** (shifted windows) via `timm`, **SegFormer** (Hugging Face) pour la segmentation, **CNN vs ViT** (accuracy / vitesse / données), attention maps | DermaMNIST (classif fine-grained), dataset de segmentation |
| 12 | `12_deploiement_torchscript_onnx.ipynb` | Sérialisation **TorchScript** (`trace`/`script`), export **ONNX**, inférence ONNX Runtime, quantization/optimisation, pipeline de bout en bout (prétraitement → inférence) | Réutilise un modèle entraîné (TP 04/10/11) |
| 13 | `13_projet_final.ipynb` | **Projet fil rouge** par équipe de 2–3 : pipeline DL complet sur un cas réel, initié dès le Module 1 et enrichi progressivement | Au choix de l'équipe |

## Objectifs pédagogiques (cf. `plan.md`)

- Maîtriser les **architectures GAN** et leurs applications (synthèse, super-résolution,
  traduction de domaine, augmentation de données).
- Comprendre et implémenter les **Vision Transformers** (ViT, Swin) pour la classification et
  la **segmentation sémantique** (SegFormer), et savoir les comparer aux CNN.
- **Déployer** un pipeline ML complet, de la préparation des données à l'inférence
  (**TorchScript / ONNX**).
- Réaliser un **projet fil rouge** intégrant l'ensemble des compétences acquises.

## Projet fil rouge (TP 13)

Chaque équipe choisit un use case réel et livre un pipeline complet. Thèmes proposés
(`plan.md`) :

- CV médicale : classification / segmentation de pathologies (IRM cérébrale, rétinopathie).
- Contrôle qualité industriel : détection/localisation de défauts (CNN + CAM).
- Maintenance prédictive : estimation **RUL** turbofan (NASA C-MAPSS) avec LSTM + CNN 1D.
- Super-résolution satellite (SRGAN ou Diffusion sur Sentinel-2).
- Segmentation de scènes urbaines (SegFormer sur Cityscapes).
- Génération d'images médicales par diffusion (DDPM conditionnel).
- Audit **XAI** complet (Grad-CAM + SHAP) sur un modèle médical.

**Critères d'évaluation** : pertinence et justification du choix architectural ; qualité du
pipeline (prétraitement, augmentation, validation) ; métriques adaptées au use case
(accuracy, mAP, IoU, SSIM, FID…) ; clarté de la présentation et démo live.

## Datasets

**Aucun dataset commité.** Les TP 10–12 réutilisent autant que possible le fil rouge médical
(**DermaMNIST** via `medmnist`) et les modèles déjà entraînés ; le projet (TP 13) utilise le
dataset propre à chaque équipe.

## Pré-requis & installation (prévisionnel)

- Modules 1 à 3 terminés.
- Dépendances envisagées :

```bash
pip install torch torchvision medmnist timm transformers numpy matplotlib scikit-learn
# déploiement :
pip install onnx onnxruntime
```

Un GPU CUDA est fortement recommandé (entraînement GAN et fine-tuning ViT).

## Convention checkpoints

À produire dans `./checkpoints/`, nommage `<arch>_<dataset>_best.pt`
(ex. `dcgan_celeba_best.pt`, `vit_dermamnist_best.pt`), plus les artefacts d'export
`*.torchscript.pt` / `*.onnx` pour le TP de déploiement.

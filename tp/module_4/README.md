# Module 4 — GAN, Vision Transformers, Déploiement & Projet final

**Formation avancée en Deep Learning avec PyTorch — Computer Vision & Séries temporelles**
*Bassem Ben Hamed — ENETCOM, Université de Sfax (`bassem.benhamed@enetcom.usf.tn`)*

> 🚧 **En cours de développement.** Les **TP 10 (GAN), 11 (Vision Transformers) et
> 12 (déploiement) sont disponibles** ; le TP 13 (projet fil rouge) décrit ci-dessous est
> prévu, conformément à `plan.md` (Module 4) et aux conventions du dépôt.

Module final : les **GAN** et leurs variantes, les **Vision Transformers**, la **mise en
production** (TorchScript / ONNX), puis le **projet fil rouge** intégrant l'ensemble des
compétences de la formation.

## Notebooks

La numérotation continue celle des modules précédents (01–09).

| # | Notebook | Thèmes | Données |
|---|----------|--------|---------|
| 10 | `10_gan_dcgan_cgan_pix2pix_srgan.ipynb` | Jeu minimax (D\* et JSD), loss **non saturante**, boucle adversariale from scratch (`BCEWithLogitsLoss`, deux optimiseurs), **DCGAN** (recettes Radford, interpolation latente), **cGAN** (`nn.Embedding` des labels, augmentation des classes rares), **WGAN-GP** (critique, `autograd.grad`, gradient penalty vérifiée analytiquement), **Pix2Pix** (colorisation Y→RGB, U-Net + PatchGAN, adv + λ·L1), **SRGAN** ×2 (blocs résiduels, `PixelShuffle`, loss perceptuelle via le ResNet18 du TP 04, PSNR/SSIM, compromis perception–distorsion) | DermaMNIST 28×28 (GPU/MPS recommandé) |
| 11 | `11_vision_transformers_vit_swin_segformer.ipynb` | **Patch embedding** *from scratch* (`Conv2d` ≡ `nn.Unfold`), **self-attention** (formule + exemple numérique) & **multi-têtes** validées contre `F.scaled_dot_product_attention`, **ViT** (`[CLS]`, encodage positionnel, blocs pré-norm) entraîné + **attention rollout**, **Swin** *from scratch* (W-MSA / SW-MSA via `torch.roll` + masques, biais de position relatif, complexité 64×, patch merging), **SegFormer** *from scratch* (attention à séquence réduite, **Mix-FFN**, décodeur tout-MLP) pour la **segmentation** de lésions (pseudo-masques Otsu, **Dice / IoU**), **CNN vs ViT** (macro-F1 / paramètres / débit), écosystème `timm` / Hugging Face *(gardé `HAS_*`)* | DermaMNIST (classification + segmentation) |
| 12 | `12_deploiement_torchscript_onnx.ipynb` | Pourquoi exporter (**artefact autonome** vs `state_dict`), **TorchScript** `trace` vs `script` (+ piège du **contrôle de flux figé**, démontré), export **ONNX** (`opset_version`, `dynamic_axes`) & inférence **ONNX Runtime** (batch variable), **validation numérique** (10⁻⁵), **quantification affine *from scratch*** (échelle $s$ / zéro-point $z$, exemple chiffré du cours $s=2/255$, par tenseur vs par canal), **dynamique** & **statique PTQ** (`fuse_model` / calibration / `convert`, *qnnpack*) avec **gains réels** (taille ÷4, latence, macro-F1), **pipeline de bout en bout** (prétraitement → inférence), artefacts `*.torchscript.pt` / `*.onnx` / `*_int8.pt` validés | Réutilise le **ResNet18/DermaMNIST du TP 04** (repli : entraînement express) |
| 13 | `13_projet_final.ipynb` *(prévu)* | **Projet fil rouge** par équipe de 2–3 : pipeline DL complet sur un cas réel, initié dès le Module 1 et enrichi progressivement | Au choix de l'équipe |

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
(ex. `dcgan_dermamnist_best.pt`, `srgan_dermamnist_best.pt`, `vit_dermamnist_best.pt`), plus les artefacts d'export
`*.torchscript.pt` / `*.onnx` pour le TP de déploiement.

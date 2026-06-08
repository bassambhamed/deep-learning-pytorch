# Module 3 — Autoencoders, VAE & Modèles de Diffusion

**Formation avancée en Deep Learning avec PyTorch — Computer Vision & Séries temporelles**
*Bassem Ben Hamed — ENETCOM, Université de Sfax (`bassem.benhamed@enetcom.usf.tn`)*

> 🚧 **En cours de développement.** Ce dossier est pour l'instant vide ; ce README décrit le
> contenu prévu, conforme à `plan.md` (Module 3) et aux conventions du dépôt. Les notebooks
> seront ajoutés au fil du développement.

Travaux pratiques sur les **modèles génératifs** : de l'autoencodeur déterministe au VAE
probabiliste, jusqu'aux modèles de diffusion (état de l'art en génération d'images).

## Notebooks prévus

La numérotation continue celle des modules précédents (01–06).

| # | Notebook (prévu) | Thèmes | Données envisagées |
|---|------------------|--------|--------------------|
| 07 | `07_autoencoders_debruitage.ipynb` | Architecture encodeur-décodeur (`nn.Module`), loss de reconstruction MSE, **ConvAE** (`nn.ConvTranspose2d` / `Upsample`+`Conv2d`), **Denoising AE**, skip connections (style U-Net), espace latent + UMAP | Images médicales (DermaMNIST / type IRM-CT), séries capteurs |
| 08 | `08_vae_variational_autoencoder.ipynb` | **Reparametrization trick** (`z = μ + ε·σ`), **KL divergence**, **ELBO**, **β-VAE** (disentanglement), **Conditional VAE**, interpolation latente, génération de séries synthétiques | DermaMNIST, séries temporelles (augmentation) |
| 09 | `09_diffusion_ddpm_ddim.ipynb` | **DDPM** (forward/reverse process, noise schedule, `loss = MSE(ε, ε_θ(x_t,t))`), **U-Net conditionné par le timestep** (GroupNorm + SiLU), **DDIM** sampling accéléré, comparaison GAN vs Diffusion | Images médicales / petites images |

## Objectifs pédagogiques (cf. `plan.md`)

- Concevoir et entraîner des **Autoencoders** (AE dense, ConvAE) pour la **compression**, le
  **débruitage** et la **détection d'anomalies**.
- Comprendre les **Variational Autoencoders** et leur **espace latent structuré** (ELBO, β-VAE).
- Comprendre et implémenter les **modèles de diffusion** (DDPM) pour la génération d'images
  haute qualité, et le **sampling DDIM** accéléré (×10–50 sans réentraînement).

## Datasets

Comme dans les modules vision précédents, **aucun dataset n'est commité** : tout est
téléchargé automatiquement et mis en cache dans `./data/`. Le fil rouge médical se poursuit
sur **DermaMNIST** (via `medmnist`) ; les tâches de débruitage/diffusion peuvent aussi
illustrer l'imagerie médicale (IRM/CT) et les séries temporelles industrielles.

## Pré-requis & installation (prévisionnel)

- Modules 1 et 2 terminés (PyTorch, MLP, CNN, transfer learning).
- Dépendances envisagées :

```bash
pip install torch torchvision medmnist numpy matplotlib scikit-learn umap-learn
# prototypage diffusion (optionnel) :
pip install diffusers
```

Un GPU CUDA est fortement recommandé pour le TP de diffusion (entraînement du U-Net).

## Convention checkpoints

À produire dans `./checkpoints/`, nommage `<arch>_<dataset>_best.pt`
(ex. `convae_dermamnist_best.pt`, `vae_dermamnist_best.pt`, `ddpm_unet_best.pt`).

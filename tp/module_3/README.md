# Module 3 — Autoencoders, VAE & Modèles de Diffusion

**Formation avancée en Deep Learning avec PyTorch — Computer Vision & Séries temporelles**
*Bassem Ben Hamed — ENETCOM, Université de Sfax (`bassem.benhamed@enetcom.usf.tn`)*

Les trois TP du module — **07** (autoencoders), **08** (VAE) et **09** (diffusion) —
sont disponibles, conformément à `plan.md` (Module 3) et aux conventions du dépôt.

Travaux pratiques sur les **modèles génératifs** : de l'autoencodeur déterministe au VAE
probabiliste, jusqu'aux modèles de diffusion (état de l'art en génération d'images).

## Notebooks

La numérotation continue celle des modules précédents (01–06).

| # | Notebook | Thèmes | Données |
|---|----------|--------|---------|
| 07 | `07_autoencoders_debruitage.ipynb` | Architecture encodeur-décodeur (`nn.Module`), loss de reconstruction MSE, **ConvAE** (`nn.ConvTranspose2d` / `Upsample`+`Conv2d`), **Denoising AE**, skip connections (style U-Net), espace latent + UMAP | Images médicales (DermaMNIST / type IRM-CT), séries capteurs |
| 08 | `08_vae_variational_autoencoder.ipynb` | **Reparametrization trick** (`z = μ + ε·σ`), **KL divergence**, **ELBO**, **β-VAE** (disentanglement), **Conditional VAE**, interpolation latente (lerp/slerp), **VAE 1D** — génération de séries capteurs synthétiques | DermaMNIST, signaux capteurs simulés (augmentation) |
| 09 | `09_diffusion_ddpm_ddim.ipynb` | **DDPM** (forward/reverse process, noise schedule linéaire vs cosinus, `loss = MSE(ε, ε_θ(x_t,t))`), **U-Net conditionné par le timestep** (time embedding sinusoïdal, GroupNorm + SiLU, self-attention), **DDIM** sampling accéléré ×20 déterministe, comparaison AE/VAE/GAN/Diffusion, prototypage `diffusers` | DermaMNIST 28×28 (GPU recommandé pour 64×64) |

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

## Pré-requis & installation

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
(ex. `convae_dermamnist_best.pt`, `vae_dermamnist_best.pt`, `ddpm_unet_dermamnist_best.pt`).

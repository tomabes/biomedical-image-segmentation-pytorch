# Biomedical Image Segmentation with PyTorch

Deep learning pipeline for binary segmentation of biomedical images from several organs and imaging modalities. The final model uses **SegFormer with an ImageNet-pretrained MiT-B2 encoder** and was developed for the OpenBio image-segmentation competition.

[Open the final notebook in Google Colab](https://colab.research.google.com/drive/1kIU4ndTVYtsdVejdR8hsJuy39YEdMrwh#scrollTo=lkcpkPRH0giN)

## Approach

- SegFormer with a MiT-B2 encoder, implemented with `segmentation-models-pytorch`;
- Dice + binary cross-entropy loss for imbalanced foreground masks;
- weighted sampling to reduce imbalance between organ groups;
- Albumentations-based spatial and intensity augmentation;
- mixed-precision training, AdamW, learning-rate scheduling and early stopping;
- patient-level train/validation splitting and test-set inference with RLE submission generation.

## Results

The best checkpoint reached **0.7539 validation IoU**. The final notebook also records a **0.7386 competition score**. These values refer to different evaluation sets and are reported separately.

![Training curves](assets/training-curves.png)

![Prediction overlays](assets/prediction-overlays.png)

The final notebook also runs inference for 144 unlabeled test images and produces encoded masks for submission.

## Repository structure

```text
notebooks/
  Toma_2026_OpenBio_competition.ipynb
assets/
  training-curves.png
  prediction-overlays.png
requirements.txt
```

## Run the project

Google Colab with a GPU runtime is recommended. Open the notebook using the link above, provide the competition data when prompted and run the cells in order. For a local environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/Toma_2026_OpenBio_competition.ipynb
```

The original dataset and trained weights are not committed to this repository.

## Tools

Python, PyTorch, SegFormer, MiT-B2, `segmentation-models-pytorch`, Albumentations, OpenCV, NumPy, pandas and Matplotlib.

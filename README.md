# Biomedical Image Segmentation with PyTorch

Deep learning pipeline for binary segmentation of biomedical images from several organs and imaging modalities. The final model uses **DeepLabV3+ with a ResNet-50 encoder** and was developed for the OpenBio image-segmentation competition.

[Open the final notebook in Google Colab](https://colab.research.google.com/drive/1kIU4ndTVYtsdVejdR8hsJuy39YEdMrwh#scrollTo=lkcpkPRH0giN)

## Approach

- DeepLabV3+ implemented with `segmentation-models-pytorch`;
- Dice + binary cross-entropy loss for imbalanced foreground masks;
- weighted sampling to reduce imbalance between organ groups;
- Albumentations-based spatial and intensity augmentation;
- mixed-precision training, AdamW, learning-rate scheduling and early stopping;
- threshold tuning and test-set inference with submission generation.

## Results

The best checkpoint reached **0.6334 validation IoU**. The competition submission scored **0.73 Jaccard** on the external hidden test set. These values refer to different evaluation sets and are reported separately.

![Training curves](assets/training-curves.png)

![Prediction overlays](assets/prediction-overlays.png)

The final notebook also runs inference for 144 unlabeled test images and produces encoded masks for submission.

## Repository structure

```text
notebooks/
  biomedical_image_segmentation_deeplabv3plus.ipynb
assets/
  training-curves.png
  prediction-overlays.png
legacy/
  openbio_unet_resnet50_colab_export.py
requirements.txt
```

`legacy/` contains an earlier Colab export based on a U-Net-style model with a ResNet-50 encoder. It is kept to show the project iteration; the notebook in `notebooks/` is the final version.

## Run the project

Google Colab with a GPU runtime is recommended. Open the notebook using the link above, provide the competition data when prompted and run the cells in order. For a local environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/biomedical_image_segmentation_deeplabv3plus.ipynb
```

The original dataset and trained weights are not committed to this repository.

## Tools

Python, PyTorch, DeepLabV3+, ResNet-50, `segmentation-models-pytorch`, Albumentations, OpenCV, NumPy, pandas and Matplotlib.

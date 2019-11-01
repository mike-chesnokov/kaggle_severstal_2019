# kaggle_severstal_2019

Parts of my python/ pytorch code from [Kaggle Severstal Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection) competition 2019.

## Description of competition
The task of this segmentation competition was to localize and classify surface defects on a steel sheet images. There were 12k training images (~6k with defects and ~6k without), 4 classes of defects. Labeled ground truth masks of defects [were very noisy](https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/113891).

## Files
- `05_1fold_experiments.ipynb` - pipeline for 1 fold experiments;
- `05_5folds_experiments.ipynb` - pipeline for 5 fold experiments;
- `trainers.py` - Trainer Class for segmentation 1fold and cross-validation;
- `trainers_classification.py` - Trainer class for multilabel classificaiton 1fold and cross-validation;
- `samplers.py` - Pytorch and my custom Samplers to sample images from dataset, including: SubsetSequentSampler, SubsetRandomSampler, ClassProbSampler;
- `losses.py` - several losses types for training, including: BCE, Dice, Focal, Tversky. Also they class weighted variants and combinations (ex. BCE-Dice);
- `datasets.py` - Pytorch datasets for segmentation and multilable classification;
- `meter.py` - Class for computing and monitoring of metrics, should be refactored;
- `utils.py` - visualization, seeds, rle coding methods, etc.;
- `configs.py` - Some constansts and lists of images to exclude from training.

# Single Image Test-Time Adaptation For Segmentation

#### By  [Haoyu Dong](https://scholar.google.com/citations?user=eZVEUCIAAAAJ&hl=en), [Nicholas Konz](https://nickk124.github.io/), [Hanxue Gu](https://scholar.google.com/citations?user=aGjCpQUAAAAJ&hl=en) and [Maciej Mazurowski](https://sites.duke.edu/mazurowski/).

[![arXiv Paper](https://img.shields.io/badge/arXiv-2402.09604-orange.svg?style=flat)](https://arxiv.org/abs/2402.09604)

<img src='https://github.com/mazurowski-lab/single-image-test-time-adaptation/blob/main/BNstats.png' width='100%'>

This is the code for our CVPR 2024 DEF-AI-MIA Workshop paper [**Medical Image Segmentation with InTEnt: Integrated Entropy Weighting for Single Image Test-Time Adaptation**](https://arxiv.org/abs/2402.09604), where we investigate an extreme case of Test-Time Adaptation (TTA) in which only a single test image is available, i.e., Single Image TTA, or **SITTA**. In this setting, we do not (and cannot) rely on the assumption that test data have balanced class distribution, balanced label distribution, and are available in large quantities. This is especially applicable to the medical imaging fieldm where data acquisition is hard and expensive. 

We introduce a method for the SITTA of segmentation models, **InTEnt**, which works by integrating over predictions made with various estimates of target domain statistics between the training and test statistics, weighted based on their entropy statistics. We validated our method on 24 source/target domain splits across 3 medical image datasets, and it surpasses the leading SITTA method by 2.9% Dice coefficient on average.

To use our codebase, we provide (a) pre-trained models for algorithm development and (b) codes to train your own models.

## 1) Use Pre-trained Models
We provide model checkpoints for the [GMSC dataset](http://cmictig.cs.ucl.ac.uk/niftyweb/challenge/), with "site2" being the source domain [here](https://drive.google.com/file/d/1fe9M6Zf2p_6SqjTWy8XNrNdSszMLVq3E/view?usp=sharing). Once put these checkpoints in the desired folder ("../checkpoints"), you can verify the reported performance in the paper with 
```
python3 main_sitta.py --dataset gmsc --phase 2
```

## 2) Train Your Own Models

### Data Preparation
Please put your images in the following format:
```
IMAGE_FOLDER:
├── domain1_aaa.png
├── domain1_bbb.png
├── domain1_ccc.png
├── ...
├── domain2_aaa.png
├── domain2_bbb.png
├── domain2_ccc.png
├── ...

MASK_FOLDER:
├── domain1_aaa.png
├── domain1_bbb.png
├── domain1_ccc.png
├── ...
├── domain2_aaa.png
├── domain2_bbb.png
├── domain2_ccc.png
├── ...
```

### Training
To train your own models, you can use the following code:

```
python3 main_supervised.py --dataset custom --image_dir IMAGE_FOLDER --mask_dir MASK_FOLDER --domain 1
```

## Citation

Please cite our paper if you use our code or reference our work:
```bib
@inproceedings{Dong2024MedicalIS,
  title={Medical Image Segmentation with InTEnt: Integrated Entropy Weighting for Single Image Test-Time Adaptation},
  author={Haoyu Dong and N. Konz and Han Gu and Maciej A. Mazurowski},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:267682146}
}
```


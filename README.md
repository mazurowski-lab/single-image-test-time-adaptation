# Single Image Test-Time Adaptation For Segmentation

#### By  [Haoyu Dong](https://scholar.google.com/citations?user=eZVEUCIAAAAJ&hl=en), [Nicholas Konz](https://nickk124.github.io/), Hanxue Gu and [Maciej Mazurowski](https://sites.duke.edu/mazurowski/).

[![arXiv Paper]()

This is the code for our paper [**"Medical Image Segmentation with InTEnt: Integrated Entropy Weighting for Single Image Test-Time Adaptation"**](), where we investigate an extreme case of Test-Time Adaptation (TTA) in which only a single test image is available, i.e., Single Image TTA, or **SITTA**. 

In this setting, we do not (and cannot) rely on the assumption that test data have balanced class distribution, balanced label distribution, and are available in large quantities. Moreover, the application of our method is easy, especially in the medical image field where data acquisition is hard and expensive. 

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

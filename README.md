# GroupRegNet
Implementation of GroupRegNet: A Groupwise One-shot Deep Learning-based 4D Image Registration Method. 

GroupRegNet is an unsupervised deep learning-based DIR method that employs both groupwise registration and one-shot strategy to register 4D medical images and then to determine all pairwise deformation vector fields (DVFs). 

## Requirement

- PyTorch
- SimpleITK: read mhd files
- logging and tqdm

## Required Data

To evaluate GroupRegNet with `registration_dirlab.py`, the [DIR-Lab](https://www.dir-lab.com/index.html) dataset is required. The original data needs to be converted into mhd format. 

## Overall structure

![groupreg_flowchart](images/groupreg_flowchart.png)

## Result

![res_1](images/res_1.png)


![res_2](images/res_2.png)


![res_3](images/res_3.png)

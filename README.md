# Foresee What You Will Learn: Data Augmentation for Domain Generalization in Non-Stationary Environments

[__[Paper]__](https://arxiv.org/pdf/2301.07845.pdf) 
&nbsp; 
This is the authors' official PyTorch implementation for Directional Data Augmentation (DDA) method in the **AAAI 2023** paper [Foresee What You Will Learn: Data Augmentation for Domain Generalization in Non-Stationary Environments](https://arxiv.org/pdf/2301.07845.pdf).


## Prerequisites
- PyTorch >= 1.12.1 (with suitable CUDA and CuDNN version)
- torchvision >= 0.10.0
- torchmeta >= 1.8.0
- Python3
- Numpy
- pandas

## Dataset
Rotated Gaussian and Rotated MNIST: [https://drive.google.com/file/d/1o80mLQcMHej9d-MznWjGp48QRBCyWTX9/view?usp=sharing](https://drive.google.com/file/d/1o80mLQcMHej9d-MznWjGp48QRBCyWTX9/view?usp=sharing)


## Training
Rotated Gaussian experiment
```
python scripts/train.py --data_dir=../dataset --gpu 0 --algorithm DDA --dataset EDGEvolCircle  --test_env 29 --steps 5001 --hparams "{\"batch_size\":120}"
```

Rotated MNIST experiment
```
python scripts/train.py --data_dir=../dataset --gpu 0 --algorithm DDA --dataset EDGRotatedMNIST --test_env 8 --steps 5001 --hparams "{\"env_number\":9}"

```



## Acknowledgement
This code is implemented based on the [domainbed](https://github.com/facebookresearch/DomainBed) code.

## Citation
If you use this code for your research, please consider citing:
```
@article{zeng2023foresee,
  title={Foresee What You Will Learn: Data Augmentation for Domain Generalization in Non-Stationary Environments},
  author={Zeng, Qiuhao and Wang, Wei and Zhou, Fan and Ling, Charles and Wang, Boyu},
  journal={arXiv preprint arXiv:2301.07845},
  year={2023}
}
```
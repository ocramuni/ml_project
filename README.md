# Reproducing CNN2

**Progetto Machine Learning**, A. A. 2024/2025

Written by Marco Giunta

## Installazione

### Download dataset
  * Scaricare i dataset smallNORB e ModelNet2D usati nell'articolo. [Link](https://drive.google.com/open?id=1S47qOBWZtSA4emTQNCR3mft6lUyJB4ke)
  * Per scaricare il dataset RGB-D eseguire il comando:
```shell
bash gather-rgbd-dataset.sh
```

### Librerie
  * Installare [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)
  * Per eseguire il codice su una GPU Nvidia, creare una _conda environment_ con i comandi:

```shell
source ~/miniconda3/bin/activate
conda env create -p ./envs -f environment_gpu.yml
conda activate ./envs
```
  * oppure, per usare solo la CPU, usare i comandi:
```shell
source ~/miniconda3/bin/activate
conda env create -p ./envs -f environment_cpu.yml
conda activate ./envs
```

## Valutare le prestazioni del progetto
Eseguire i notebook `cnn2_modelnet_py.ipynb`, `cnn2_smallnorb_pt.ipynb` e `cnn2_rgbd_pt.ipynb` per valute le prestazioni del progetto.
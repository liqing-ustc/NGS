# Neural-Grammar-Symbolic Learning with Back-Search
Pytorch implementation for Neural-Grammar-Symbolic Learning with Back-Search (NGS-BS) on the Handwritten Formula Recognition task.

<div align="center">
  <img src="RL_vs_BS.png" width="750px">
</div>

### Publication
**[Closed Loop Neural-Symbolic Learning via Integrating Neural Perception, Grammar Parsing, and Symbolic Reasoning](https://liqing-ustc.github.io/NGS/)**
<br>
[Qing Li](http://liqing-ustc.github.io/), 
[Siyuan Huang](http://siyuanhuang.com/), 
[Yining Hong](https://evelinehong.github.io/), 
[Yixin Chen](https://yixchen.github.io/), 
[Ying Nian Wu](http://www.stat.ucla.edu/~ywu/), and
[Song-Chun Zhu](http://www.stat.ucla.edu/~sczhu/)
<br>
In International Conference on Machine Learning (*ICML*) 2020.
<br>

```
@inproceedings{li2020ngs,
  title={Closed Loop Neural-Symbolic Learning via Integrating Neural Perception, Grammar Parsing, and Symbolic Reasoning},
  author={Li, Qing and Huang, Siyuan and Hong, Yining and Chen, Yixin and Wu, Ying Nian and Zhu, Song-Chun.},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2020}
}
```

## Prerequisites
* Ubuntu 16.04
* Python 3.7
* NVIDIA TITAN RTX + CUDA 10.0
* PyTorch 1.4.0

## Getting started
1. Download the Handwritten Formula Recognition dataset from [google drive](https://drive.google.com/file/d/1G07kw-wK-rqbg_85tuB7FNfA49q8lvoy/view?usp=sharing) and unzip it:
```
unzip HWF.zip
```
2. Create an environment with all packages from `requirements.txt` installed (Note: please double check the CUDA version on your machine and install pytorch accordingly):
```
conda create -y -n ngs python=3.7
source activate ngs
pip install -r requirements.txt
```

## Train the models
To reproduce the experiment results, we can simply run the following code:
```
sh run.sh
```
This script will train different model variants and save the training logs into the `output` directory. Since it will take a long time for the RL and MAPO baselines to converge, we put the final training logs in the `results` folder.

## Plot the training curves
To plot the training curves, run:
```
jupyter notebook
```
Open the `plot_results.ipynb` and run all cells.

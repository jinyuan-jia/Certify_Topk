This repository contains code for "Certified Robustness for Top-k Predictions against Adversarial Perturbations via Randomized Smoothing". 

Required python tool: numpy, scipy, statsmodels. 

# Code usage: 

topk.py is used to compute the certified radius for top-k predictions. 

compute_radius.py is used to read the frequency file (each line contains the frequency of each label for a sample and last column is the label that we aim to certify, "cifar0.25.txt" is an example file) and save the results. 

You can directly run "python3 run.py" 

We also run the code and "result.txt" in the result file we obtained. 

# Citation 

If you use this code, please cite the following paper: 

```
@inproceedings{
jia2020certified,
title={Certified Robustness for Top-k Predictions against Adversarial Perturbations via Randomized Smoothing},
author={Jinyuan Jia and Xiaoyu Cao and Binghui Wang and Neil Zhenqiang Gong},
booktitle={International Conference on Learning Representations},
year={2020}
}
```

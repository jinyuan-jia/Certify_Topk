This repository contains code for our ICLR 2020 paper "Certified Robustness for Top-k Predictions against Adversarial Perturbations via Randomized Smoothing".

Required python tool: numpy, scipy, statsmodels. 

# Code usage: 

topk.py is used to compute the certified radius for top-k predictions. 

compute_radius.py is used to read the frequency file (each line contains the frequency of each label for an example, i.e., the frequencies of labels 1,2,...,c. Last column contains the label that we aim to certify. "cifar0.25.txt" is an example file) and save the results. 

Given a base classifier _f_, Gaussian noise &epsilon; ~ N(0,&sigma;I<sup>2</sup>), and an example _x_, we sample _n_ random noise from N(0,&sigma;I<sup>2</sup>), i.e., &epsilon;<sub>1</sub>, &epsilon;<sub>2</sub>, ... , &epsilon;<sub>_n_</sub>. The frequency for label _l_ can be computed as n<sub>_l_</sub> = &sum;<sub>i</sub> _Indicate_( _f_( _x_ + &epsilon;<sub>i</sub> ) = _l_ ), where _Indicate_ is indicate function. 

You can directly run:
``` python3 compute_radius.py --src cifar0.25.txt --dst result.txt --alpha 0.001 --sigma 0.25 --k 1 ``` 
where result.txt is file that saves the result, which contains two columns. The first column contains the example id and the second column contains the certified radius. alpha, sigma, and k specifies the value of &alpha;, &sigma;, and k in the paper (please refer to paper for details). When estimating the upper and lower bounds for the probabilities, we adopt <b>SimuEM</b> (please refer to our paper for details).

We also ran the code and "result.txt" is the result file we obtained. 

Note that when we conducted the experiments, we used the pre-trained models from previous work which can be found in this address: https://github.com/locuslab/smoothing

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

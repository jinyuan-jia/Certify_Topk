from __future__ import print_function
import numpy as np
from statsmodels.stats.proportion import proportion_confint, multinomial_proportions_confint
from scipy.stats import norm, binom_test
from topk import binary_search_compute_r,CertifyRadius
import argparse


parser = argparse.ArgumentParser(description='Certify from saved frequency file')
parser.add_argument("--src", type=str, help="source file for the sample")
parser.add_argument("--dst", type=str, help="file to save the results", default=None)
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--sigma", type=float, default=1.0, help="noise hyperparameter")
parser.add_argument("--k", type=int, default=1, help="top k parameter")
args = parser.parse_args()

def multi_ci(counts, alpha):
    multi_list = []
    n = np.sum(counts)
    l = len(counts)
    for i in range(l): 
        multi_list.append(proportion_confint(min(max(counts[i], 1e-10), n-1e-10), n, alpha=alpha*2./l, method="beta"))
    return np.array(multi_list)

if __name__ == "__main__":

    data = np.genfromtxt(args.src, dtype=int)
    
    print("idx\t radius", flush=True)  

    num_classes = data.shape[-1] - 1
    num_data = data.shape[0]
    certified_r = []
    
    if args.dst is not None:
        f = open(args.dst, 'w')
        print("idx\tradius", file=f, flush=True)
    
    for idx in range(data.shape[0]):
        ls = data[idx][-1]

        class_freq = data[idx][:-1]
        CI = multi_ci(class_freq, args.alpha)
        pABar = CI[ls][0]
        probability_bar = CI[:,1]
        probability_bar = np.clip(probability_bar, a_min=-1, a_max=1-pABar)
        probability_bar[ls] = pABar
        r = CertifyRadius(ls, probability_bar, args.k, args.sigma)
       
        print("{}\t{:.3}".format(idx+1, r), flush=True)
        if args.dst is not None:
            print("{}\t{:.3}".format(idx+1, r), file=f, flush=True)
     
    if args.dst is not None:
        f.close()
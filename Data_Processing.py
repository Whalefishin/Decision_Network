import sys
import math
import collections

def main():
    W_N_W = open("W_N_W_ult.txt")
    W_N_N = open("W_N_N_ult.txt")
    W_N_diff = open("W_N_diff_ult.txt")
    W_N_Acc = open("W_N_Acc_ult.txt")
    W_N_RT = open("W_N_RT_ult.txt")

    W_N_W_vector = []
    W_N_N_vector = []
    W_N_diff_vector = []

    base_name_W_N = "W_N"
    gain_type = ["Binary","Sigm"]
    sep_gain = ["IntAll", "Sep"]
    normalization = ["YesNorm","NoNorm"]
    biased_IC = ["FairIC","UnfairIC"]

    for g,gain in enumerate(gain_type):
        for s,sep in enumerate(sep_gain):
            for n,norm in enumerate(normalization):
                for b,biased in enumerate(biased_IC):
                    


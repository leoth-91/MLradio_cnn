#!/bin/bash
source activate tensorflow
#conda list -n tensorflow
conda info -e

#python /run_CNN/CNN_parallel.py --tag=CNN_notnorm --tag_res=run2_ocprl --path=/archive/home/sammazza/radioML/ --path_TD=/archive/home/sammazza/radioML/data/mapsim_PS/ --path_TL=/archive/home/sammazza/radioML/data/mapsim_PS/ --N_start=0 --N_stop=14999 --N_epochs_small=15 --N_epochs_large=15 --batch_size_small=4 --batch_size_large=4 --train --refined --pool --suffix=x1000

python /run_CNN/CNN_parallel.py --tag=CNN_N_b --tag_res=run1_prl_noise --path=/archive/home/sammazza/radioML/ --path_TD=/archive/home/sammazza/radioML/data/mapsim_PS_N_b/ --path_TL=/archive/home/sammazza/radioML/data/mapsim_PS_N_b/ --N_start=0 --N_stop=14999 --N_epochs_small=15 --N_epochs_large=15 --batch_size_small=6 --batch_size_large=6 --train --refined --pool --suffix=x1000

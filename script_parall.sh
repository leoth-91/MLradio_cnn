#!/bin/bash
source activate tensorflow
#conda list -n tensorflow
conda info -e

python /run_CNN/CNN_parallel.py --tag=CNN_notnorm --tag_res=run1_ocprl --path=/archive/home/sammazza/radioML/ --path_TD=/archive/home/sammazza/radioML/data/mapsim_PS/ --path_TL=/archive/home/sammazza/radioML/data/mapsim_PS/ --N_start=0 --N_stop=10000 --N_epochs_small=1 --N_epochs_large=1 --batch_size_small=8 --batch_size_large=8 --train --suffix=x1000

from matplotlib import pyplot as plt
import numpy as np
import argparse
from matplotlib import rcParams
from matplotlib.ticker import ScalarFormatter, LogFormatter, LogFormatterSciNotation, LogFormatterMathtext



rcParams['font.family'] = 'serif'
rcParams.update({'text.usetex': True})
rcParams.update({'figure.autolayout': True})

parser = argparse.ArgumentParser()
parser.add_argument("--tag_res", type=str, help="tag of the network results", default='')
parser.add_argument("--path", type=str, help="general path of results", default='/home/simone/RadioML/results/')
parser.add_argument("--path_dest", type=str, help="path of plots", default='/home/simone/RadioML/plots/')
parser.add_argument("--N_plot", type=int, help="number of plots to produce (default: 4)", default=4)
parser.add_argument("--linear", action='store_true', help="Plot on linear scale")

args = parser.parse_args()
tag_res = args.tag_res
path = args.path
path_dest = args.path_dest
N_plot = args.N_plot
linear = args.linear

if tag_res is not '':
    tag_res = '_' + tag_res



print('Creating Loss Plot:')
dpi = 300
try:
    loss = np.transpose(np.genfromtxt(path+'/loss_function.txt', dtype=np.float32))

    plt.plot(loss[0], loss[1], '.-')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    # plt.ylim((0,np.max(loss[1])*1.1))
    plt.ylim((np.min(loss[1]),np.max(loss[1])*1.01))
    # plt.ylim((0,0.015))
    plt.xlim((np.min(loss[0]), np.max(loss[0])*1.01))
    # plt.grid()
    plt.savefig(path_dest+'/0_loss.png', dpi=dpi)
    plt.close()
except Exception as e:
    print(e)

print('Done.')


# List of all colors:
# https://matplotlib.org/3.1.0/gallery/color/named_colors.html
color_pred_small = 'dodgerblue'
color_pred_full = 'limegreen'
color_label = 'dimgray'

# List of different markers:
# https://matplotlib.org/api/markers_api.html


# some experimentation...
# class LogFormatterTexTextMode(LogFormatter):
#     def __call__(self, x, pos=None):
#         x = LogFormatter.__call__(self, x,pos)
#         s = r"10\textsuperscript{{{}}}".format(x)
#         return s


# This file stores the displayed relative difference between target and prediction in a txt-file
with open(path_dest+'percentage_variations.txt','w') as stats:
    stats.write('#Map-number  small-scale      large-scale\n')

for i in range(N_plot):
    # plt.figure(figsize=(12,10))
    map_num = str(i).zfill(5)
    data_small = np.genfromtxt(path+'_small/'+'2-PCF_map_'+map_num+tag_res+'_parall_small.txt')
    data_large = np.genfromtxt(path+'_large/'+'2-PCF_map_'+map_num+tag_res+'_parall_large.txt')

    # Pathing together the targets to have them as a single connected curve
    thetas = np.concatenate((data_small[:,0], data_large[:,0]))
    target = np.concatenate((data_small[:,2], data_large[:,2]))

    print('Percentage variation of map number {:}:'.format(map_num))
    variation_small = np.round(np.abs(1-data_small[:,1]/data_small[:,2])*100, 2)
    variation_large = np.round(np.abs(1-data_large[:,1]/data_large[:,2])*100, 2)
    print(variation_small, ' and ', variation_large)
    print('Overall variation: {:}'.format(np.round(sum(variation_small)+sum(variation_large)), 3))

    # storing the values as file...
    with open(path_dest+'percentage_variations.txt','a') as stats:
        stats.write('{:}    {:}     {:}\n'.format(map_num, variation_small, variation_large))

    plt.plot(thetas,target, '.-', color=color_label, label='Target function', linewidth=2, markersize=12)
    plt.plot(data_small[:,0],data_small[:,1],   '+', color=color_pred_full, label='Prediction small scale', linewidth=4, markersize=15, markeredgewidth=2)
    plt.plot(data_large[:,0],data_large[:,1], '+', color=color_pred_small,label='Prediction large scale', linewidth=4, markersize=15, markeredgewidth=2)

    # Plot dividing :ine
    # loc: is the theta-value that lies between the smallest large-scale and the biggest small-scale values

    # loc = np.max(data_small[:,0]) + (np.min(data_large[:,0]) - np.max(data_small[:,0]))/2
    # plt.axvline(x=loc, ymin=0, ymax=1, color='darkgrey',linewidth=4,markersize=18)


    plt.xlabel(r'$\theta\,\mathrm{[deg]}$',fontsize=15)
    plt.ylabel(r'$\xi(\theta)\times10^3$',fontsize=15)

    if linear == False:
        plt.xscale('log')
        plt.yscale('log')

    plt.tick_params(length=5, which='major', labelsize=15, pad=7)#,length=6,width=3)
    plt.tick_params(length=5, which='minor', labelsize=15, pad=7)#,length=6,width=3)
    
    # ax = plt.gca()
    # ax.yaxis.set_major_formatter(LogFormatter())
    # ax.xaxis.set_major_formatter(LogFormatter())

    # plt.legend(loc='upper right', fontsize=12, framealpha=0.5, fancybox=True)
    plt.legend(loc='lower left', fontsize=12, framealpha=0.5, fancybox=True)
    plt.savefig(path_dest+'2-PCF_map_'+str(i).zfill(5)+tag_res+'.png', dpi=dpi)
    #plt.show()
    plt.clf()











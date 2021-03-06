from matplotlib import pyplot as plt
import numpy as np
import argparse
from matplotlib import rcParams
from matplotlib.ticker import ScalarFormatter, LogFormatter, LogFormatterSciNotation, LogFormatterMathtext
from scipy.interpolate import  interp1d
from scipy.integrate import quad
from scipy.special import eval_legendre as Pl
from scipy.stats import binned_statistic
from scipy.special import eval_legendre as leg

rcParams['font.family'] = 'serif'
rcParams.update({'text.usetex': True})
rcParams.update({'figure.autolayout': True})

parser = argparse.ArgumentParser()
parser.add_argument("--tag_res", type=str, help="tag of the network results", default='')
parser.add_argument("--path", type=str, help="general path of input files and results", default='/home/simone/RadioML/')
parser.add_argument("--path_dest", type=str, help="path of plots", default='/home/simone/RadioML/plots/')
parser.add_argument("--label_file", type=str, help="APS file", default='/home/simone/RadioML/data/Cl_CNN_0_10000_label.dat')
parser.add_argument("--N_plot", type=int, help="number of plots to produce (default: 1)", default=1)
parser.add_argument("--N_begin", type=int, help="matching with testing subset (default: 9500)", default=9500)
parser.add_argument("--linear", action='store_true', help="Plot on linear scale")
parser.add_argument("--fact", type=float, help="normalization factor for the correlation function (default = 1000)", default=1000)

args = parser.parse_args()
tag_res = args.tag_res
path = args.path
path_dest = args.path_dest
N_plot = args.N_plot
linear = args.linear
label_file = args.label_file
fact = args.fact #factor used to normalize CCF
N_begin = args.N_begin #Cl matching CCF testing
theta_min = 0.01
theta_max = 2

if tag_res is not '':
    tag_res = '_' + tag_res

# List of all colors:
# https://matplotlib.org/3.1.0/gallery/color/named_colors.html
color_pred_small = 'dodgerblue'
color_pred_full = 'limegreen'
#color_pred_full = 'forestgreen'
#color_pred_full = 'orange'
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
    stats.write('#Map-number   total     small-scale      large-scale\n')

ll = np.genfromtxt(label_file,usecols=0)
cl_label = np.genfromtxt(label_file)[:,N_begin:]

th_list = np.logspace(np.log10(theta_min), np.log10(theta_max), num=10)
A = (2.*ll+1.)**2/(4.*np.pi)**2   #normalization for circular simmetry
#A_sqrd = A**2               #normalization for circular simmetry
Pl_list=[]
for th in th_list:
   Pl_list.append(leg(ll,np.cos(np.radians(th)))**2)
Pl_list=np.array(Pl_list)

for i in range(N_plot):
    # plt.figure(figsize=(12,10))
    map_num = str(i).zfill(5)
    data_small = np.genfromtxt(path+'results_small/'+'2-PCF_map_'+map_num+tag_res+'_small.txt')
    data_large = np.genfromtxt(path+'results_large/'+'2-PCF_map_'+map_num+tag_res+'_large.txt')

    # Pathing together the targets to have them as a single connected curve
    thetas = np.concatenate((data_small[:,0], data_large[:,0]))
    target = np.concatenate((data_small[:,2], data_large[:,2]))

    print('Percentage variation of map number {:}:'.format(map_num))
    variation_small = np.round(np.abs(1-data_small[:,1]/data_small[:,2])*100, 2)
    variation_large = np.round(np.abs(1-data_large[:,1]/data_large[:,2])*100, 2)
    print(variation_small, ' and ', variation_large)
    total_var = np.round(sum(variation_small)+sum(variation_large), 3)
    print('Overall variation: {:}'.format(total_var))

    theta_tot = np.concatenate((data_small[:,0],data_large[:,0]))
    data_tot = np.concatenate((data_small[:,1],data_large[:,1]))

    theta_tot = np.concatenate((np.array([0.]),theta_tot))
    data_tot = np.concatenate((np.array([data_tot[0]]),data_tot))
    f = interp1d(np.radians(theta_tot),data_tot)

    sig2 = []
    for tt in range(len(th_list)):
        sig2.append(2*np.sum(A*Pl_list[tt]*(cl_label[:,i]*fact)**2))

    # storing the values as file...
    with open(path_dest+'percentage_variations.txt','a') as stats:
        stats.write('{:}    {:}     {:}     {:}\n'.format(map_num, total_var, variation_small, variation_large))

    #plt.plot(thetas,target, '.-', color=color_label, label='Target function', linewidth=2, markersize=12)
    (_,caps,_) = plt.errorbar(thetas,target, yerr=np.sqrt(sig2), fmt='o-', color=color_label, label='Target function', linewidth=2, markersize=2, capsize=10,)
    for cap in caps:
        cap.set_markeredgewidth(2)
    plt.plot(data_small[:,0],data_small[:,1], '+', color=color_pred_full, label='Prediction small scale', linewidth=4, markersize=15, markeredgewidth=2)
    plt.plot(data_large[:,0],data_large[:,1], '+', color=color_pred_small,label='Prediction large scale', linewidth=4, markersize=15, markeredgewidth=2)

    # plt.ylim((0.1, 20))

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

    plt.legend(loc='upper right', fontsize=12, framealpha=0.5, fancybox=True)
    #plt.legend(loc='lower left', fontsize=12, framealpha=0.5, fancybox=True)
    plt.savefig(path_dest+'check_gauss_err_'+str(i).zfill(5)+tag_res+'.pdf')
    plt.show()
    plt.clf()
print('Done.')

from matplotlib import pyplot as plt
import numpy as np
import argparse
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams.update({'text.usetex': True})
rcParams.update({'figure.autolayout': True})

parser = argparse.ArgumentParser()
parser.add_argument("--tag_res", type=str, help="tag of the network results", default='')
parser.add_argument("--path", type=str, help="general path of results", default='/home/simone/RadioML/results/')
parser.add_argument("--path_dest", type=str, help="path of plots", default='/home/simone/RadioML/plots/')
parser.add_argument("--N_plot", type=int, help="number of plots to produce (default: 4)", default=4)

args = parser.parse_args()
tag_res = args.tag_res
path = args.path
path_dest = args.path_dest
N_plot = args.N_plot

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


# colors
# https://matplotlib.org/3.1.0/gallery/color/named_colors.html
color_pred_small = 'dodgerblue'
# color_pred_full = 'red'
color_pred_full = 'limegreen'
color_label = 'black'

# markers:
# https://matplotlib.org/api/markers_api.html


for i in range(N_plot):
    # plt.figure(figsize=(12,10))
    data_full = np.genfromtxt(path+'_full/'+'2-PCF_map_'+str(i).zfill(5)+tag_res+'.txt')
    data_small = np.genfromtxt(path+'_small/'+'2-PCF_map_'+str(i).zfill(5)+tag_res+'.txt')

    # Pathing together the targets to have them as a single connected curve
    thetas = np.concatenate((data_full[:,0], data_small[:,0]))
    target = np.concatenate((data_full[:,2], data_small[:,2]))

    print('Percentage variation:')
    print(np.round(np.abs(1-data_full[:,1]/data_full[:,2])*100,1), ' and ', np.round(np.abs(1-data_small[:,1]/data_small[:,2])*100,1))


    plt.plot(thetas,target, '.-', color=color_label,label='Target function',linewidth=2,markersize=12)

    plt.plot(data_full[:,0],data_full[:,1],   '+', color=color_pred_full, label='Prediction small scale',linewidth=4,markersize=15, markeredgewidth=2)
    plt.plot(data_small[:,0],data_small[:,1], '+', color=color_pred_small,label='Prediction large scale',linewidth=4,markersize=15, markeredgewidth=2)


    # Plot division line
    # loc: is the theta-value that lies between the smallest large-scale and the biggest small-scale values
    # loc = np.max(data_full[:,0]) + (np.min(data_small[:,0]) - np.max(data_full[:,0]))/2
    # plt.axvline(x=loc, ymin=0, ymax=1, color='darkgrey',linewidth=4,markersize=18)


    plt.xlabel(r'$\theta \, \mathrm{[deg]}$',fontsize=15)
    plt.ylabel(r'$\xi \, (\theta) \, \mathrm{X 10^3}$',fontsize=15)
    plt.xscale('log')
    plt.yscale('log')
    # plt.tick_params(direction='in', width=2, length=5, axis='both', which='major', labelsize=10, pad=7)#,length=6,width=3)
    # plt.tick_params(direction='in', width=2, length=5, axis='both', which='minor', labelsize=10, pad=7)#,length=6,width=3)
    plt.tick_params(direction='in', length=5, axis='both', which='major', labelsize=15, pad=7)#,length=6,width=3)
    plt.tick_params(direction='in', length=5, axis='both', which='minor', labelsize=15, pad=7)#,length=6,width=3)
    plt.legend(loc='upper right',fontsize=12,framealpha=0.5, fancybox=True)

    plt.savefig(path_dest+'2-PCF_map_'+str(i).zfill(5)+tag_res+'.png', dpi=dpi)
    #plt.show()
    plt.clf()










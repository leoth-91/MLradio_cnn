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

def calculations(target, prediction):
    data = np.stack((target, prediction, np.zeros(len(target)), np.zeros(len(target)), np.zeros(len(target))), axis=1)
    # data[:,0] = target
    # data[:,1] = pred
    # later: data[:,2] = bin number
    # later: data[:,3] = std-dev of the target values in the correspinding bin
    # later: data[:,4] = differnce between target and prediction in std-devs
    ##############
    # CALCULATIONS
    # Defining the bins
    target_min = np.min(data[:,0])
    target_max = np.max(data[:,0])

    N_bins = 10
    delta = (target_max - target_min)/N_bins

    # now find out to which bin each target belongs
    for i in range(len(data[:,1])):    # loop through all data points
        for j in range(N_bins):
            if data[i,1] > target_min+j*delta and data[i,1] < target_min+(j+1)*delta:
                data[i,2] = j

    # calculating the number of values in each bin
    # to later calculate the mean value of each bin
    num_bins = []
    for j in range(N_bins): # loop through all bins
        count = 0
        for k in range(len(data[:,1])):
            if data[k,2] == j:
                count += 1
        num_bins.append(count) 
    # calculating the mean in each bin:
    mean_bins = []
    for j in range(N_bins): # loop through all bins
        mean = 0
        for k in range(len(data[:,1])):
            if data[k,2] == j:
                mean += data[k,0]/num_bins[j]
        mean_bins.append(mean)
    # calculate the std variation for each bin
    stds = []
    for j in range(N_bins):
        temp = []
        for k in range(len(data[:,1])):
            if data[k,2] == j:
                temp.append(data[k,0])
        stds.append(np.std(temp))
    # store the bins srd-deviation in the data file
    # print(stds)
    for j in range(N_bins): # loop through all bins
        for k in range(len(data[:,1])):
            if data[k,2] == j:
                data[k,3] = stds[j]
    # print(data[:,3]) # if values here are zer0 that probably means that there is no 

    for j in range(N_bins): # loop through all bins
        for k in range(len(data[:,1])):
            if data[k,2] == j:
                # data[k,4] = |target - prediction|/std_dev
                data[k,4] = np.floor(np.abs(data[k,0] - data[k,1])/(data[k,3]+1E-7))
    # print(data[:,4])


    return data



def visualization(data, name='image.png', theta=666):
    ##########
    # PLOTTING
    c1 = 'tab:green'
    c2 = 'orange'
    c3 = 'red'
    ct = 'mediumblue'  # good: mediumblue/blue
    # List of all available colors: https://matplotlib.org/3.1.0/gallery/color/named_colors.html
    ms = 5
    lw = 2

    err_count = 0
    for k in range(len(data[:,1])):
        num_sig = int(data[k,4])
        if num_sig > 100:
            err_count += 1

    # this is done with 3 single loops as I want the 1-sigma points to overlap the 2-sigma points and those to overlap the others
    for k in range(len(data[:,1])):
        num_sig = int(data[k,4])
        if num_sig >1 and num_sig< 100:
            plt.plot(data[k,1],data[k,0], '.', color=c3, label=None, linewidth=lw, markersize=ms)
    for k in range(len(data[:,1])):
        num_sig = int(data[k,4])
        if num_sig == 1:
            plt.plot(data[k,1],data[k,0], '.', color=c2, label=None, linewidth=lw, markersize=ms)
    for k in range(len(data[:,1])):
        num_sig = int(data[k,4])
        if num_sig == 0:
            plt.plot(data[k,1],data[k,0], '.', color=c1, label=None, linewidth=lw, markersize=ms)

    # empty plots to have the labels listed in the legend
    plt.plot([],[], '.', color=c1,  label=r'within $1\sigma$', linewidth=lw, markersize=ms+3)
    plt.plot([],[], '.', color=c2, label=r'within $2\sigma$', linewidth=lw, markersize=ms+3)
    plt.plot([],[], '.', color=c3,    label=r'$>2\sigma$ error', linewidth=lw, markersize=ms+3)

    # Plot ideal prediction line
    max = np.max((np.max(data[:,0]), np.max(data[:,1])))
    ideal = np.linspace(0, max, num=10)
    plt.plot(ideal, ideal, '-', color=ct, label='ideal prediction', linewidth=1, markersize=1)
    # plt.plot(data[:,0], data[:,0], '-', color=ct, label='ideal prediction', linewidth=1, markersize=1)
    
    # plt.xlabel(r'$\xi_\mathrm{pred.}/\bar\xi_\mathrm{true}\ '+r'(\theta={:.2})$'.format(theta),fontsize=15)
    plt.xlabel(r'$\xi_\mathrm{pred.}/\bar\xi_\mathrm{true}$',fontsize=15)
    plt.ylabel(r'$\xi_\mathrm{true}/ \bar\xi_\mathrm{true}$',fontsize=15)

    plt.xlim((0, max))
    plt.ylim((0, max))

    plt.legend(title=r'Values at $\theta = {:.2}\,^\circ$'.format(theta), loc='lower right', fontsize=12, framealpha=0.5, fancybox=True, shadow=False)
    
    # Error count is the number of values that have 0 std-deviation (due to only one element in bin) 
    # (and hence unreasonable high difference between target and prediction when expressed in std-deviations)
    # If the validation set is sufficiently large (in the hundreds) this almost never happens, if it does, it is ignored
    print('Error count is {:}/{:} ({:.2}%)'.format(err_count, len(data[:,0]), err_count/len(data[:,0])*100))

    plt.savefig(name, dpi=300)
    plt.clf()


########################################
# load all data:
all_preds   = []
all_targets = []
for i in range(N_plot):
    # load data from file
    map_num = str(i).zfill(5)
    data_small = np.genfromtxt(path+'_small/'+'2-PCF_map_'+map_num+tag_res+'_parall_small.txt')
    data_large = np.genfromtxt(path+'_large/'+'2-PCF_map_'+map_num+tag_res+'_parall_large.txt')

    # concatenate small/large scale togther
    theta  = np.concatenate((data_small[:,0], data_large[:,0]))
    pred   = np.concatenate((data_small[:,1], data_large[:,1]))
    target = np.concatenate((data_small[:,2], data_large[:,2]))

    # concatenate to create a vector with all prediction/target values
    all_preds.append(pred)
    all_targets.append(target)

all_targets = np.array(all_targets)
all_preds = np.array(all_preds)

for k in range(all_targets.shape[1]):
    pred = all_preds[:,k]
    target = all_targets[:,k]
    if True:
        pred /= np.mean(target)
        target /= np.mean(target)

    data = calculations(target, pred)

    visualization(data, name=path_dest+'visualization_test-set_xi_'+str(k+1).zfill(2)+'.png', theta=theta[k])


# create multiplot from created pngs

from PIL import Image
list_im = []
for k in range(all_targets.shape[1]):
    list_im.append(path_dest+'visualization_test-set_xi_'+str(k+1).zfill(2)+'.png')

all_images = [ Image.open(i) for i in list_im ]

if False:
    left  = np.concatenate([np.asarray(all_images[i]) for i in range(5)], axis=0)
    right = np.concatenate([np.asarray(all_images[i+5]) for i in range(5)], axis=0)
    full = np.concatenate((left,right), axis=1)
else:
    row1 = np.concatenate((np.asarray(all_images[0]), np.asarray(all_images[1])), axis=1)
    row2 = np.concatenate((np.asarray(all_images[2]), np.asarray(all_images[3])), axis=1)
    row3 = np.concatenate((np.asarray(all_images[4]), np.asarray(all_images[5])), axis=1)
    row4 = np.concatenate((np.asarray(all_images[6]), np.asarray(all_images[7])), axis=1)
    row5 = np.concatenate((np.asarray(all_images[8]), np.asarray(all_images[9])), axis=1)

    full = np.concatenate((row1,row2,row3,row4,row5), axis=0)

full = Image.fromarray(full)
full.save(path_dest+'all.png')




















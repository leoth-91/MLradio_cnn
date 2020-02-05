#!/usr/bin/env python

import numpy as np
import glob
from PIL import Image
import h5py
import argparse
import time

from utility import image_provider


parser = argparse.ArgumentParser()
parser.add_argument("--tag", type=str, help="tag of the data run", default='')
parser.add_argument("--tag_res", type=str, help="tag of the network results", default='')
parser.add_argument("--path", type=str, help="general path of CNN and test results", default='')
parser.add_argument("--path_TD", type=str, help="path of training data (.tif files)", default='')
parser.add_argument("--path_TL", type=str, help="path of training label (correlation function)", default='')
parser.add_argument("--N_start", type=int, help="start number of the maps", default=0)
parser.add_argument("--N_stop", type=int, help="end number of the maps", default=0)
parser.add_argument("--max_ID", type=int, help="maximum number of maps", default=-1)
parser.add_argument("--N_epochs_full", type=int, help="number of epochs (default = 10)", default=10)
parser.add_argument("--N_epochs_small", type=int, help="number of epochs (default = 10)", default=10)
parser.add_argument("--LR_full", type=float, help="learning rate full", default=1.e-5)
parser.add_argument("--LR_small", type=float, help="learning rate small", default=1.e-5)
parser.add_argument("--batch_size_full", type=int, help="batch_size (default = 10)", default=10)
parser.add_argument("--batch_size_small", type=int, help="batch_size (default = 10)", default=10)
parser.add_argument("--kernel_size", type=int, help="kernel_size (default = 5)", default=5)
parser.add_argument("--pool_size", type=int, help="pool_size (default = 4)", default=4)
parser.add_argument("--stride", type=int, help="stride (default = 1)", default=1)
parser.add_argument("--train", action='store_true', help="train the CNN")
parser.add_argument("--refined", action='store_true', help="using different kernel size for each layer")
parser.add_argument("--norm_label", action='store_true', help="normalization of labels")
parser.add_argument("--norm_data", action='store_true', help="normalization of data")
parser.add_argument("--suffix", type=str, help="suffix for label file.", default='')

args = parser.parse_args()
tag = args.tag
tag_res = args.tag_res
path = args.path


path_train_data_full = args.path_TD + 'full/'
path_train_data_small = args.path_TD + 'small/'

path_train_label_full = args.path_TL + 'full/'
path_train_label_small = args.path_TL + 'small/'


N_start = args.N_start
N_stop = args.N_stop
train = args.train
LR_full = args.LR_full
LR_small = args.LR_small

norm_label = args.norm_label
norm_data = args.norm_data
max_ID = args.max_ID
suffix = args.suffix

refined = args.refined
if refined:
    print('Using refined network.')
    from utility import network_refined as network
else:
    print('Using base network.')
    from utility import network

if tag is not '':
   tag = tag+'_'

if tag_res is not '':
   tag_res = '_' + tag_res

if suffix is not '':
   suffix = '_' + suffix

# paths of images and path of the spectra.dat-file
path_train_label_full = path_train_label_full+'CCF_'+tag+str(N_start)+'_'+str(N_stop)+'_label'+suffix+'.dat'
path_train_label_small = path_train_label_small+'CCF_'+tag+str(N_start)+'_'+str(N_stop)+'_label'+suffix+'.dat'

path_model = path+'saved_model/'

path_results_full = path+'results_full/'
path_results_small = path+'results_small/'

##########################################################
## Parameters ############################################
N_epochs_full = args.N_epochs_full
N_epochs_small = args.N_epochs_small
batch_size_full = args.batch_size_full
batch_size_small = args.batch_size_small

KS = args.kernel_size
PS = args.pool_size
stride = args.stride

model_parameters_full = {'learning_rate': LR_full,      # 1E-5
                       'decay_rate': LR_full,      # 1E-5 # i.e. lr /= (1+decay_rate) after each epoch
                      'kernel_size': (KS,KS),
                        'pool_size': (PS,PS),
                           'stride': stride
                    }
model_parameters_small = {'learning_rate': LR_small,      # 1E-5
                       'decay_rate': LR_small,      # 1E-5 # i.e. lr /= (1+decay_rate) after each epoch
                      'kernel_size': (KS,KS),
                        'pool_size': (PS,PS),
                           'stride': stride
                    }
##########################################################
##########################################################
def emptyDirectory(thePath):
    import os, shutil
    for the_file in os.listdir(thePath):
        file_path = os.path.join(thePath, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)
##########################################################
## Setting up the Generators #############################
#from keras.backend.tensorflow_backend import set_session
#import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)
#sess = tf.Session(config=config)
#set_session(sess)  # set this TensorFlow session as the default session for Keras

# Creating the 'partition' dictionary that contains all the image IDs ('msim_0000_data' etc.)
# divided into a training, validation and test set

all_IDs_full = []  # to store all IDs
all_IDs_small = []  # to store all IDs

N_files_full = len(glob.glob(path_train_data_full+'*_data.tif'))
N_files_small = len(glob.glob(path_train_data_small+'*_data.tif'))

for i in range(N_files_full):
    all_IDs_full.append('msim_'+tag+'%04d_data'%(i))
for i in range(N_files_small):
    all_IDs_small.append('msim_'+tag+'%04d_data'%(i))

all_IDs_full = all_IDs_full[:max_ID]
all_IDs_small = all_IDs_small[:max_ID]

# Splitting into training/validation/test
num1_full = int(len(all_IDs_full)-0.15*len(all_IDs_full))
num2_full = int(len(all_IDs_full)-0.05*len(all_IDs_full))
num1_small = int(len(all_IDs_small)-0.15*len(all_IDs_small))
num2_small = int(len(all_IDs_small)-0.05*len(all_IDs_small))

training_IDs_full = all_IDs_full[:num1_full]
validation_IDs_full = all_IDs_full[num1_full:num2_full]
test_IDs_full = all_IDs_full[num2_full:]

training_IDs_small = all_IDs_small[:num1_small]
validation_IDs_small = all_IDs_small[num1_small:num2_small]
test_IDs_small = all_IDs_small[num2_small:]


partition_full = {'train': training_IDs_full,
             'validation': validation_IDs_full,
             'test': test_IDs_full}
partition_small = {'train': training_IDs_small,
             'validation': validation_IDs_small,
             'test': test_IDs_small}

print("Full size dataset:")
print('Training with %i/%i images'
        %(len(partition_full['train']), len(partition_full['train'])+len(partition_full['validation'])+len(partition_full['test'])))
print('Validating on %i/%i images'
        %(len(partition_full['validation']), len(partition_full['train'])+len(partition_full['validation'])+len(partition_full['test'])))
print('Testing on %i/%i images'
        %(len(partition_full['test']), len(partition_full['train'])+len(partition_full['validation'])+len(partition_full['test'])))

print("Small size dataset:")
print('Training with %i/%i images'
        %(len(partition_small['train']), len(partition_small['train'])+len(partition_small['validation'])+len(partition_small['test'])))
print('Validating on %i/%i images'
        %(len(partition_small['validation']), len(partition_small['train'])+len(partition_small['validation'])+len(partition_small['test'])))
print('Testing on %i/%i images'
        %(len(partition_small['test']), len(partition_small['train'])+len(partition_small['validation'])+len(partition_small['test'])))

# Reading the spectra.dat-file and store all spectra
all_labels_full = np.transpose(np.genfromtxt(path_train_label_full, dtype=np.float32)[:,1:max_ID])
all_labels_small = np.transpose(np.genfromtxt(path_train_label_small, dtype=np.float32)[:,1:max_ID])

if len(all_labels_full)!=len(all_IDs_full):
   print('lenght labels:  lenght data:')
   print(len(all_labels_full),len(all_IDs_full))
   print('ERROR: labels not matching training set.')
   exit()
if len(all_labels_small)!=len(all_IDs_small):
   print('lenght labels:  lenght data:')
   print(len(all_labels_small),len(all_IDs_small))
   print('ERROR: labels not matching training set.')
   exit()

# norming the labels to values from 0...y_max
if norm_label:
    y_max = 1.0
    max_value_label_full = np.max(all_labels_full)
    all_labels_full = all_labels_full*y_max
    all_labels_full = all_labels_full/max_value_label

    max_value_label_small = np.max(all_labels_small)
    all_labels_small = all_labels_small*y_max
    all_labels_small = all_labels_small/max_value_label
    
# Creating a dictionary that associates the right correlation function to each ID
# labels = {'msim_0000_data': [0.082, 0.20930, ....]],
#           'msim_0001_data': [0.082, 0.20930, ....]],
#           ....}




labels_full = {}
for k,label in enumerate(all_IDs_full):
    labels_full[label] = all_labels_full[k]

# Number of output nodes has to be the number of points in the correlation function
#N_out = len(labels['msim_'+tag+'0000_data'])
N_out_full = len(all_labels_full[0])

# Read a single image in order to determine the pixel-size
image_full = np.array(Image.open(path_train_data_full+partition_full['train'][0]+'.tif'))

n_x_full = image_full.shape[0]    # the shorter side
n_y_full = image_full.shape[1]
n_channels_full = 1

# data shape for the input of the CNN
data_shape_full = (n_x_full, n_y_full, n_channels_full)





labels_small = {}
for k,label in enumerate(all_IDs_small):
    labels_small[label] = all_labels_small[k]

# Number of output nodes has to be the number of points in the correlation function
#N_out = len(labels['msim_'+tag+'0000_data'])
N_out_small = len(all_labels_small[0])

# Read a single image in order to determine the pixel-size
image_small = np.array(Image.open(path_train_data_small+partition_small['train'][0]+'.tif'))

n_x_small = image_small.shape[0]    # the shorter side
n_y_small = image_small.shape[1]
n_channels_small = 1

# data shape for the input of the CNN
data_shape_small = (n_x_small, n_y_small, n_channels_small)






# parameters for the data-generators
generator_parameters_full = {'path_data': path_train_data_full,
              'dim': (n_x_full, n_y_full),
              'N_out': N_out_full,
              'batch_size': batch_size_full,
              'norm': norm_data}
generator_parameters_small = {'path_data': path_train_data_small,
              'dim': (n_x_small, n_y_small),
              'N_out': N_out_small,
              'batch_size': batch_size_small,
              'norm': norm_data}


# Definitions of the generators
print('Definition of generators...')
print(generator_parameters_full)
training_generator_full   = image_provider.DataGenerator(partition_full['train'], labels_full, shuffle=True, **generator_parameters_full)
validation_generator_full = image_provider.DataGenerator(partition_full['validation'], labels_full, shuffle=True, **generator_parameters_full)
test_generator_full = image_provider.DataGenerator(partition_full['test'], labels_full, shuffle=False, **generator_parameters_full)


# Definitions of the generators
print('Definition of generators...')
print(generator_parameters_small)
training_generator_small   = image_provider.DataGenerator(partition_small['train'], labels_small, shuffle=True, **generator_parameters_small)
validation_generator_small = image_provider.DataGenerator(partition_small['validation'], labels_small, shuffle=True, **generator_parameters_small)
test_generator_small = image_provider.DataGenerator(partition_small['test'], labels_small, shuffle=False, **generator_parameters_small)



























##########################################################
## Model and training on FULL ############################

# Defining the learning model
print('Initializing model...')
print(model_parameters_full)
model_full = network.CNN(N_out_full, data_shape=data_shape_full, **model_parameters_full)

if train==True:
    print(model_full.summary())
    print('Fitting model...')
    # parameters fed into the fit-method
    time_start = time.time()
    fit_parameters_full = {'generator': training_generator_full,
                      'validation_data': validation_generator_full,
                      'epochs': N_epochs_full}
    # training the model
    history_full = model_full.fit_generator(**fit_parameters_full)

    model.save(path_model+'model'+tag_res+'_full.h5')
    model.save_weights(path_model+'weights'+tag_res+'_full.csv')

    # Creating a file to store the loss function:
    epochs_full = np.array(range(1,1+N_epochs_full))
    loss_values_full = history_full.history['loss']

    #emptyDirectory(path_results)

    with open(path_model+'loss_function'+tag_res+'_full.txt','w') as stats:
        stats.write('#Epoch  Loss\n')

    for k in range(len(epochs)):
        with open(path_model+'loss_function'+tag_res+'.txt_full','a') as stats:
            stats.write('{:}    {:}\n'.format(epochs[k], loss_values[k]))
    print('Elapsed time:',time.time()-time_start,'s')
else:
    from keras.models import load_model
    print('Loading weights...')
    model_full = load_model(path_model+'model'+tag_res+'_full.h5')
    model_full.load_weights(path_model+'weights'+tag_res+'_full.csv')
    print(model_full.summary())



##########################################################
## Model and training on SMALL ###########################

# Defining the learning model
print('Initializing model...')
print(model_parameters_small)
model_small = network.CNN(N_out_small, data_shape=data_shape_small, **model_parameters_small)

if train==True:
    print(model_small.summary())
    print('Fitting model...')
    # parameters fed into the fit-method
    time_start = time.time()
    fit_parameters_small = {'generator': training_generator_small,
                      'validation_data': validation_generator_small,
                      'epochs': N_epochs_small}
    # training the model
    history_small = model_small.fit_generator(**fit_parameters_small)

    model.save(path_model+'model'+tag_res+'_small.h5')
    model.save_weights(path_model+'weights'+tag_res+'_small.csv')

    # Creating a file to store the loss function:
    epochs_small = np.array(range(1,1+N_epochs_small))
    loss_values_small = history_small.history['loss']

    #emptyDirectory(path_results)

    with open(path_model+'loss_function'+tag_res+'_small.txt','w') as stats:
        stats.write('#Epoch  Loss\n')

    for k in range(len(epochs)):
        with open(path_model+'loss_function'+tag_res+'.txt_small','a') as stats:
            stats.write('{:}    {:}\n'.format(epochs[k], loss_values[k]))
    print('Elapsed time:',time.time()-time_start,'s')
else:
    from keras.models import load_model
    print('Loading weights...')
    model_small = load_model(path_model+'model'+tag_res+'_small.h5')
    model_small.load_weights(path_model+'weights'+tag_res+'_small.csv')
    print(model_small.summary())







## Testing ###########################################
thetas_full = np.transpose(np.genfromtxt(path_train_label_full, dtype=np.float32)[:,0])
print('Running prediction:')
pred_full = model_full.predict_generator(test_generator_full, verbose=1)
target_full = np.asarray([*test_generator_full.labels.values()])[num2_full:]     # the * unpacks the dictionary_values-type

for k in range(target_full.shape[0]-target_full.shape[0]%batch_size_full):
    # printing the outputs
    with open(path_results_full+'2-PCF_map_'+str(k).zfill(5)+tag_res+'_full.txt','w') as stats_full:
        stats_full.write('#theta  pred    target\n')

    for i in range(len(thetas)):
        with open(path_results_full+'2-PCF_map_'+str(k).zfill(5)+tag_res+'_full.txt','a') as stats_full:
            if norm_label:
                stats_full.write('{:}    {:}    {:}\n'.format(thetas_full[i], pred_full[k,i]*max_value_label_full, target[k,i]*max_value_label_full))
            else:
                stats_full.write('{:}    {:}    {:}\n'.format(thetas_full[i], pred_full[k,i], target_full[k,i]))
    stats_full.close()


## Testing ###########################################
thetas_small = np.transpose(np.genfromtxt(path_train_label_small, dtype=np.float32)[:,0])
print('Running prediction:')
pred_small = model_small.predict_generator(test_generator_small, verbose=1)
target_small = np.asarray([*test_generator_small.labels.values()])[num2_small:]     # the * unpacks the dictionary_values-type

for k in range(target_small.shape[0]-target_small.shape[0]%batch_size_small):
    # printing the outputs
    with open(path_results_small+'2-PCF_map_'+str(k).zfill(5)+tag_res+'_small.txt','w') as stats_small:
        stats_small.write('#theta  pred    target\n')

    for i in range(len(thetas)):
        with open(path_results_small+'2-PCF_map_'+str(k).zfill(5)+tag_res+'_small.txt','a') as stats_small:
            if norm_label:
                stats_small.write('{:}    {:}    {:}\n'.format(thetas_small[i], pred_small[k,i]*max_value_label_small, target[k,i]*max_value_label_small))
            else:
                stats_small.write('{:}    {:}    {:}\n'.format(thetas_small[i], pred_small[k,i], target_small[k,i]))
    stats_small.close()


print('Done.')
































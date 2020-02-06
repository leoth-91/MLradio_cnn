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
parser.add_argument("--N_split", type=int, help="index for splitting thetas (default = 5)", default=5)
parser.add_argument("--max_ID", type=int, help="maximum number of maps", default=-1)
parser.add_argument("--N_epochs_small", type=int, help="number of epochs (default = 10)", default=10)
parser.add_argument("--N_epochs_large", type=int, help="number of epochs (default = 10)", default=10)
parser.add_argument("--LR_small", type=float, help="learning rate small scale", default=1.e-5)
parser.add_argument("--LR_large", type=float, help="learning rate large scale", default=1.e-5)
parser.add_argument("--batch_size_small", type=int, help="batch_size (default = 10)", default=10)
parser.add_argument("--batch_size_large", type=int, help="batch_size (default = 10)", default=10)
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

path_train_data = args.path_TD
path_train_label = args.path_TL

path_train_data_small = path_train_data
path_train_data_large = path_train_data

path_train_label_small = path_train_label
path_train_label_large = path_train_label

N_start = args.N_start
N_stop = args.N_stop
N_split = args.N_split #starting theta
train = args.train
LR_small = args.LR_small
LR_large = args.LR_large
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
path_train_label_small = path_train_label_small+'CCF_'+tag+str(N_start)+'_'+str(N_stop)+'_label'+suffix+'.dat'
path_train_label_large = path_train_label_large+'CCF_'+tag+str(N_start)+'_'+str(N_stop)+'_label'+suffix+'.dat'

path_model = path+'saved_model/'

path_results_small = path+'results_small/'
path_results_large = path+'results_large/'

##########################################################
## Parameters ############################################
N_epochs_small = args.N_epochs_small
N_epochs_large = args.N_epochs_large
batch_size_small = args.batch_size_small
batch_size_large = args.batch_size_large

KS = args.kernel_size
PS = args.pool_size
stride = args.stride

model_parameters_small = {'learning_rate': LR_small,      # 1E-5
                       'decay_rate': LR_small,      # 1E-5 # i.e. lr /= (1+decay_rate) after each epoch
                      'kernel_size': (KS,KS),
                        'pool_size': (PS,PS),
                           'stride': stride
                    }
model_parameters_large = {'learning_rate': LR_large,      # 1E-5
                       'decay_rate': LR_large,      # 1E-5 # i.e. lr /= (1+decay_rate) after each epoch
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

all_IDs_small = []  # to store all IDs
all_IDs_large = []  # to store all IDs

N_files_small = len(glob.glob(path_train_data_small+'*_data.tif'))
N_files_large = len(glob.glob(path_train_data_large+'*_data.tif'))

for i in range(N_files_small):
    all_IDs_small.append('msim_'+tag+'%04d_data'%(i))
for i in range(N_files_large):
    all_IDs_large.append('msim_'+tag+'%04d_data'%(i))

all_IDs_small = all_IDs_small[:max_ID]
all_IDs_large = all_IDs_large[:max_ID]

# Splitting into training/validation/test
num1_small = int(len(all_IDs_small)-0.15*len(all_IDs_small))
num2_small = int(len(all_IDs_small)-0.05*len(all_IDs_small))
num1_large = int(len(all_IDs_large)-0.15*len(all_IDs_large))
num2_large = int(len(all_IDs_large)-0.05*len(all_IDs_large))

training_IDs_small = all_IDs_small[:num1_small]
validation_IDs_small = all_IDs_small[num1_small:num2_small]
test_IDs_small = all_IDs_small[num2_small:]

training_IDs_large = all_IDs_large[:num1_large]
validation_IDs_large = all_IDs_large[num1_large:num2_large]
test_IDs_large = all_IDs_large[num2_large:]


partition_small = {'train': training_IDs_small,
             'validation': validation_IDs_small,
             'test': test_IDs_small}
partition_large = {'train': training_IDs_large,
             'validation': validation_IDs_large,
             'test': test_IDs_large}

print("Small scales size dataset:")
print('Training with %i/%i images'
        %(len(partition_small['train']), len(partition_small['train'])+len(partition_small['validation'])+len(partition_small['test'])))
print('Validating on %i/%i images'
        %(len(partition_small['validation']), len(partition_small['train'])+len(partition_small['validation'])+len(partition_small['test'])))
print('Testing on %i/%i images'
        %(len(partition_small['test']), len(partition_small['train'])+len(partition_small['validation'])+len(partition_small['test'])))

print("Large scales size dataset:")
print('Training with %i/%i images'
        %(len(partition_large['train']), len(partition_large['train'])+len(partition_large['validation'])+len(partition_large['test'])))
print('Validating on %i/%i images'
        %(len(partition_large['validation']), len(partition_large['train'])+len(partition_large['validation'])+len(partition_large['test'])))
print('Testing on %i/%i images'
        %(len(partition_large['test']), len(partition_large['train'])+len(partition_large['validation'])+len(partition_large['test'])))

# Reading the spectra.dat-file and store all spectra
all_labels_small = np.transpose(np.genfromtxt(path_train_label_small, dtype=np.float32)[:N_split,1:max_ID])
all_labels_large = np.transpose(np.genfromtxt(path_train_label_large, dtype=np.float32)[N_split:,1:max_ID])

if len(all_labels_small)!=len(all_IDs_small):
   print('lenght labels:  lenght data:')
   print(len(all_labels_small),len(all_IDs_small))
   print('ERROR: labels not matching training set.')
   exit()
if len(all_labels_large)!=len(all_IDs_large):
   print('lenght labels:  lenght data:')
   print(len(all_labels_large),len(all_IDs_large))
   print('ERROR: labels not matching training set.')
   exit()

# norming the labels to values from 0...y_max
if norm_label:
    y_max = 1.0
    max_value_label_small = np.max(all_labels_small)
    all_labels_small = all_labels_small*y_max
    all_labels_small = all_labels_small/max_value_label

    max_value_label_large = np.max(all_labels_large)
    all_labels_large = all_labels_large*y_max
    all_labels_large = all_labels_large/max_value_label

# Creating a dictionary that associates the right correlation function to each ID
# labels = {'msim_0000_data': [0.082, 0.20930, ....]],
#           'msim_0001_data': [0.082, 0.20930, ....]],
#           ....}

labels_small = {}
for k,label in enumerate(all_IDs_small):
    labels_small[label] = all_labels_small[k]

# Number of output nodes has to be the number of points in the correlation function
N_out_small = len(all_labels_small[0])

# Read a single image in order to determine the pixel-size
image_small = np.array(Image.open(path_train_data_small+partition_small['train'][0]+'.tif'))

n_x_small = image_small.shape[0]    # the shorter side
n_y_small = image_small.shape[1]
n_channels_small = 1

# data shape for the input of the CNN
data_shape_small = (n_x_small, n_y_small, n_channels_small)

labels_large = {}
for k,label in enumerate(all_IDs_large):
    labels_large[label] = all_labels_large[k]

# Number of output nodes has to be the number of points in the correlation function
#N_out = len(labels['msim_'+tag+'0000_data'])
N_out_large = len(all_labels_large[0])

# Read a single image in order to determine the pixel-size
image_large = np.array(Image.open(path_train_data_large+partition_large['train'][0]+'.tif'))

n_x_large = image_large.shape[0]    # the shorter side
n_y_large = image_large.shape[1]
n_channels_large = 1

# data shape for the input of the CNN
data_shape_large = (n_x_large, n_y_large, n_channels_large)


# parameters for the data-generators
generator_parameters_small = {'path_data': path_train_data_small,
              'dim': (n_x_small, n_y_small),
              'N_out': N_out_small,
              'batch_size': batch_size_small,
              'norm': norm_data}
generator_parameters_large = {'path_data': path_train_data_large,
              'dim': (n_x_large, n_y_large),
              'N_out': N_out_large,
              'batch_size': batch_size_large,
              'norm': norm_data}


# Definitions of the generators
print('Definition of generators for small scales...')
print(generator_parameters_small)
training_generator_small   = image_provider.DataGenerator(partition_small['train'], labels_small, shuffle=True, **generator_parameters_small)
validation_generator_small = image_provider.DataGenerator(partition_small['validation'], labels_small, shuffle=True, **generator_parameters_small)
test_generator_small = image_provider.DataGenerator(partition_small['test'], labels_small, shuffle=False, **generator_parameters_small)


# Definitions of the generators
print('Definition of generators for large scales...')
print(generator_parameters_large)
training_generator_large   = image_provider.DataGenerator(partition_large['train'], labels_large, shuffle=True, **generator_parameters_large)
validation_generator_large = image_provider.DataGenerator(partition_large['validation'], labels_large, shuffle=True, **generator_parameters_large)
test_generator_large = image_provider.DataGenerator(partition_large['test'], labels_large, shuffle=False, **generator_parameters_large)

##########################################################
## Model and training on small ###########################

# Defining the learning model
print('Initializing model for small scales...')
print(model_parameters_small)
model_small = network.CNN(N_out_small, data_shape=data_shape_small, **model_parameters_small)

if train==True:
    print(model_small.summary())
    print('Fitting model for small scales...')
    # parameters fed into the fit-method
    time_start = time.time()
    fit_parameters_small = {'generator': training_generator_small,
                      'validation_data': validation_generator_small,
                      'epochs': N_epochs_small}
    print(fit_parameters_small)
    # training the model
    history_small = model_small.fit_generator(**fit_parameters_small)

    model_small.save(path_model+'model'+tag_res+'_small.h5')
    model_small.save_weights(path_model+'weights'+tag_res+'_small.csv')

    # Creating a file to store the loss function:
    epochs_small = np.array(range(1,1+N_epochs_small))
    loss_values_small = history_small.history['loss']

    #emptyDirectory(path_results)

    with open(path_model+'loss_function'+tag_res+'_small.txt','w') as stats:
        stats.write('#Epoch  Loss\n')

    for k in range(len(epochs_small)):
        with open(path_model+'loss_function'+tag_res+'.txt_small','a') as stats:
            stats.write('{:}    {:}\n'.format(epochs_small[k], loss_values_small[k]))
    print('Elapsed time:',time.time()-time_start,'s')
else:
    from keras.models import load_model
    print('Loading weights for small scales...')
    model_small = load_model(path_model+'model'+tag_res+'_small.h5')
    model_small.load_weights(path_model+'weights'+tag_res+'_small.csv')
    print(model_small.summary())

##########################################################
## Model and training on large ###########################

# Defining the learning model
print('Initializing model for large scales...')
print(model_parameters_large)
model_large = network.CNN(N_out_large, data_shape=data_shape_large, **model_parameters_large)

if train==True:
    print(model_large.summary())
    print('Fitting model for large scales...')
    # parameters fed into the fit-method
    time_start = time.time()
    fit_parameters_large = {'generator': training_generator_large,
                      'validation_data': validation_generator_large,
                      'epochs': N_epochs_large}
    # training the model
    history_large = model_large.fit_generator(**fit_parameters_large)

    model_large.save(path_model+'model'+tag_res+'_large.h5')
    model_large.save_weights(path_model+'weights'+tag_res+'_large.csv')

    # Creating a file to store the loss function:
    epochs_large = np.array(range(1,1+N_epochs_large))
    loss_values_large = history_large.history['loss']

    #emptyDirectory(path_results)

    with open(path_model+'loss_function'+tag_res+'_large.txt','w') as stats:
        stats.write('#Epoch  Loss\n')

    for k in range(len(epochs_large)):
        with open(path_model+'loss_function'+tag_res+'.txt_large','a') as stats:
            stats.write('{:}    {:}\n'.format(epochs_large[k], loss_values_large[k]))
    print('Elapsed time:',time.time()-time_start,'s')
else:
    from keras.models import load_model
    print('Loading weights for large scales...')
    model_large = load_model(path_model+'model'+tag_res+'_large.h5')
    model_large.load_weights(path_model+'weights'+tag_res+'_large.csv')
    print(model_large.summary())


## Testing ###########################################
thetas_small = np.transpose(np.genfromtxt(path_train_label_small, dtype=np.float32)[:N_split,0])
print('thetas for small scales (deg):')
print(thetas_small)
print('Running prediction for small scales:')
pred_small = model_small.predict_generator(test_generator_small, verbose=1)
target_small = np.asarray([*test_generator_small.labels.values()])[num2_small:]     # the * unpacks the dictionary_values-type

for k in range(target_small.shape[0]-target_small.shape[0]%batch_size_small):
    # printing the outputs
    with open(path_results_small+'2-PCF_map_'+str(k).zfill(5)+tag_res+'_small.txt','w') as stats_small:
        stats_small.write('#theta  pred    target\n')

    for i in range(len(thetas_small)):
        with open(path_results_small+'2-PCF_map_'+str(k).zfill(5)+tag_res+'_small.txt','a') as stats_small:
            if norm_label:
                stats_small.write('{:}    {:}    {:}\n'.format(thetas_small[i], pred_small[k,i]*max_value_label_small, target[k,i]*max_value_label_small))
            else:
                stats_small.write('{:}    {:}    {:}\n'.format(thetas_small[i], pred_small[k,i], target_small[k,i]))
    stats_small.close()


## Testing ###########################################
thetas_large = np.transpose(np.genfromtxt(path_train_label_large, dtype=np.float32)[N_split:,0])
print('thetas for large scales (deg):')
print(thetas_large)
print('Running prediction for large scales:')
pred_large = model_large.predict_generator(test_generator_large, verbose=1)
target_large = np.asarray([*test_generator_large.labels.values()])[num2_large:]     # the * unpacks the dictionary_values-type

for k in range(target_large.shape[0]-target_large.shape[0]%batch_size_large):
    # printing the outputs
    with open(path_results_large+'2-PCF_map_'+str(k).zfill(5)+tag_res+'_large.txt','w') as stats_large:
        stats_large.write('#theta  pred    target\n')

    for i in range(len(thetas_large)):
        with open(path_results_large+'2-PCF_map_'+str(k).zfill(5)+tag_res+'_large.txt','a') as stats_large:
            if norm_label:
                stats_large.write('{:}    {:}    {:}\n'.format(thetas_large[i], pred_large[k,i]*max_value_label_large, target[k,i]*max_value_label_large))
            else:
                stats_large.write('{:}    {:}    {:}\n'.format(thetas_large[i], pred_large[k,i], target_large[k,i]))
    stats_large.close()


print('Done.')

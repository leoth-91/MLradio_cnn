#!/usr/bin/env python
#
# Written by Simone Ammazzalorso
#
import healpy as hp
from healpy.visufunc import cartview
#import matplotlib as mpl
#mpl.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import interp1d#,interp2d
#from scipy.integrate import nquad
from scipy.special import eval_legendre as leg
import time
import random
import os.path
import subprocess as sub
import argparse
from PIL import Image
import glob
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename

parser = argparse.ArgumentParser()
parser.add_argument("N_start", type=int, help="start number of the map", default=0)
parser.add_argument("N_stop", type=int, help="end number of the map", default=0)
parser.add_argument("--tag", type=str, help="tag of the run", default='')
parser.add_argument("--path", type=str, help="general path", default='')
parser.add_argument("--out_dir", type=str, help="output directory", default='')
parser.add_argument("--N1h_low", type=float, help="lower normalization for 1-halo term", default=0.1)
parser.add_argument("--N1h_up", type=float, help="upper normalization for 1-halo term", default=10.0)
parser.add_argument("--N2h_low", type=float, help="lower normalization for 2-halo term", default=0.1)
parser.add_argument("--N2h_up", type=float, help="upper normalization for 2-halo term", default=10.0)
parser.add_argument("--alpha_low", type=float, help="lower power-law index for 2-halo term", default=-2.0)
parser.add_argument("--alpha_up", type=float, help="upper power-law index for 2-halo term", default=0.0)
parser.add_argument("--add_noise", action='store_true', help="apply noise term.")
parser.add_argument("--pivot_noise", type=int, help="multipole for pivot noise level", default=100)
parser.add_argument("--save_noise", action='store_true', help="save noise map.")
parser.add_argument("--N_low", type=float, help="lower normalization for noise term", default=0.5)
parser.add_argument("--N_up", type=float, help="upper normalization for noise term", default=2.0)
parser.add_argument("--add_beam", action='store_true', help="apply beam function from file(s)")
parser.add_argument("--beam_shape", type=tuple, help="the beam will be downgraded to this shape", default=None)
parser.add_argument("--beam_size", type=float, help="the beam will be downgraded to this resolution (arcsec)", default=None)
parser.add_argument("--filter", type=int, help="set the beam filter size", default=50)
parser.add_argument("--random_beam", action='store_true', help="the beam for convolution is selected randomly from the list")
parser.add_argument("--save_beam", action='store_true', help="save new beam filter")
parser.add_argument("--beam_path", type=str, help="beam folder path", default='')
parser.add_argument("--add_gauss_beam", action='store_true', help="apply gaussian beam with healpy function function.")
parser.add_argument("--b_low", type=float, help="lower normalization for gaussian beam term", default=1.0)
parser.add_argument("--b_up", type=float, help="upper normalization for gaussian beam term", default=5.0)
parser.add_argument("--theta_min", type=float, help="theta min (deg)", default=0.01)
parser.add_argument("--theta_max", type=float, help="theta max (deg)", default=2.0)
parser.add_argument("--nbins", type=int, help="number of bins for the correlation function", default=10)
parser.add_argument("--fact", type=float, help="normalization factor for the correlation function (default = 1.0)", default=1.0)
parser.add_argument("--norm_tif", action='store_true', help="apply normalization to tif files (values from 0 to 255).")
parser.add_argument("--reject_clean", action='store_true', help="option if you do not need to save the clean map.")
parser.add_argument("--reject_map", action='store_true', help="option if you do not need to save the simulated map (e.g. if NSIDE is large).")
parser.add_argument("--show_map", action='store_true', help="show maps and projections.")
parser.add_argument("--show_beam", action='store_true', help="show beams.")
parser.add_argument("-v","--verbose", action='store_true', help="verbose")
parser.add_argument("--patch_GAL", type=int, help="select a square along the Galactic plane of desired size.", default=None)
parser.add_argument("--fast_patch", action='store_true', help="fast selection of Galactic patch projection.")

args = parser.parse_args()
tag = args.tag
path = args.path
out_dir = args.out_dir
N_start = args.N_start
N_stop = args.N_stop
N1h_low = args.N1h_low
N1h_up = args.N1h_up
N2h_low = args.N2h_low
N2h_up = args.N2h_up
alpha_low = args.alpha_low
alpha_up = args.alpha_up
add_noise = args.add_noise
pivot_noise = args.pivot_noise
save_noise = args.save_noise
N_low = args.N_low
N_up = args.N_up
add_beam = args.add_beam
beam_shape = args.beam_shape
beam_size = args.beam_size
filter = args.filter
save_beam = args.save_beam
random_beam = args.random_beam
beam_path = args.beam_path
add_gauss_beam = args.add_gauss_beam
b_low = args.b_low
b_up = args.b_up
theta_min = args.theta_min
theta_max = args.theta_max
nbins = args.nbins
fact = args.fact
norm_tif = args.norm_tif
reject_clean = args.reject_clean
reject_map = args.reject_map
show_map = args.show_map
show_beam = args.show_beam
verbose = args.verbose
patch_GAL = args.patch_GAL
fast_patch = args.fast_patch

if add_beam and add_gauss_beam:
    print('ERROR: Choose one type of beam.')
    exit(-1)

if path is not '' and path[-1] is not '/':
    path = path+'/'

def normalization(moll_array):
    moll_array = moll_array + np.abs(np.min(moll_array))
    moll_array = moll_array/(np.max(moll_array))*255.0
    return moll_array

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

if add_beam:
    print('*** Loading beams...')
    beams = []
    # get all beam files from beam_path with '*.fits'
    beam_ids = glob.glob(beam_path + '*.fits')
    #print(beam_ids)
    #print(len(beam_ids))
    # loading the fits-files and stacking them together
    for k, beam_id in enumerate(beam_ids):
        print('beam ID:',beam_id)
        with fits.open(beam_id) as hdul:
            beam_data = hdul[0].data
            hdr = hdul[0].header
            old_shape = np.shape(beam_data)
            old_size = np.abs(hdr['CDELT1'])
            if hdr['CUNIT1'] == 'deg':
                old_size = old_size*3600.0
                print('resolution: {:.1f} arcsec'.format(old_size))

        beam_data = beam_data[0][0][:][:]
        print('shape:',np.shape(beam_data))
        '''
        #CHECK NORMALIZATION
        x1, x2 = hdr['CDELT1'], hdr['CDELT1']+len(beam_data)*np.abs(hdr['CDELT1'])
        xx = np.linspace(x1, x2, num=len(beam_data))
        y1, y2 = hdr['CDELT2'],hdr['CDELT2']+len(beam_data)*np.abs(hdr['CDELT2'])
        yy = np.linspace(y1, y2, num=len(beam_data))
        beam_interp = interp2d(xx,yy,beam_data)
        def wrap(x,y):
            return beam_interp(x,y)[0]
        print(beam_interp(x1,y2))
        print(wrap(x1,y2))
        result = nquad(wrap,[x1,x2],[y1,y2])
        print(result)
        exit()
        '''
        if show_beam:
            plt.imshow(beam_data)
            plt.colorbar()
            ##plt.savefig('test_beam.pdf')
            plt.show()
            plt.clf()

        if beam_size is not None:
            new_size = beam_size
            print('new resolution is set to: {:.1f} arcsec'.format(new_size))
            ns = int(len(beam_data)*old_size//new_size)
            new_shape = (ns,ns)
            print('new shape:',new_shape)
            ##ALTERNATIVE RESIZING
            #from photutils.psf.matching import resize_psf
            #new_psf = resize_psf(beam_data,2.0,46.0)
            #idx1 = (new_shape[0]-filter)//2
            #idx2 = (new_shape[1]-filter)//2
            #new_psf = new_psf[idx1:idx1+filter,idx2:idx2+filter]
            #plt.imshow(new_psf)
            #plt.colorbar()
            #plt.show()
            #plt.clf()
            #print(np.shape(new_psf))
            #print(np.sum(new_psf))
            #
        if beam_shape is not None:
            new_shape = beam_shape
            print('beam shape is set to:',beam_shape)
            new_size = len(beam_data)*old_size//new_shape[0]
            print('new resolution: {:.1f} arcsec'.format(new_size))

        if beam_data.shape is not new_shape:
            print('Downgrading beam array...')
            xdiv = beam_data.shape[0]//new_shape[0]
            ydiv = beam_data.shape[1]//new_shape[1]
            xx = (int(new_shape[0]*(xdiv+1))-beam_data.shape[0])//2
            yy = (int(new_shape[1]*(ydiv+1))-beam_data.shape[1])//2
            pad = np.zeros((xx,beam_data.shape[1]))
            beam_data = np.concatenate((beam_data,pad),axis=0)
            beam_data = np.concatenate((pad,beam_data),axis=0)
            pad = np.zeros((beam_data.shape[0],yy))
            beam_data = np.concatenate((beam_data,pad),axis=1)
            beam_data = np.concatenate((pad,beam_data),axis=1)
            beam_data = rebin(beam_data,new_shape)
            #beam_data = beam_data*(new_size/old_size)**2
            print(np.shape(beam_data))
        if filter is not None:
            print('Selecting filter...')
            idx1 = (new_shape[0]-filter)//2
            idx2 = (new_shape[1]-filter)//2
            beam_data = beam_data[idx1:idx1+filter,idx2:idx2+filter]
            print(np.shape(beam_data))
            if show_beam:
                plt.imshow(beam_data)
                plt.colorbar()
                ##plt.savefig('test_beam.pdf')
                plt.show()
                plt.clf()

        if save_beam:
            beam_id = beam_id.replace('.fits','_res{:.1f}.fits'.format(new_size))
            beam_id = beam_id.replace('.fits','_filter{:d}x{:d}.fits'.format(filter,filter))
            print(beam_id)
            #NOTE: FINISH THIS PART!
        beams.append(beam_data)

#power spectrum files; assumed to be normalized with l*(l+1)/(2 Pi)
in_1halo = 'Cl_radio_1.dat'
in_2halo = 'Cl_radio_2.dat'

###Geneal options
#Healpix size
NSIDE = 4096
#multipole range
l_start = 5
l_stop = 10000
ncl = 30
idx = np.logspace(np.log10(l_start),np.log10(l_stop-1),num=ncl,dtype=int)
#size of tif image
x_size = 20000
y_size = int(x_size/2)

plot_test = False
###############################################################
NPIX = 12*NSIDE**2
ll = np.arange(l_start,l_stop)
norm = 2.*np.pi/(ll*(ll+1))

if fast_patch:
    lon = np.linspace(0.0,360.0/x_size*patch_GAL,num=patch_GAL)
    lat = np.flip(np.linspace(-90.0/y_size*patch_GAL,90.0/y_size*patch_GAL,num=patch_GAL))
    cc = np.array(np.meshgrid(lat, lon))
    cc = np.reshape(cc.T,(-1,2))
    lat = cc[:,0]
    lon = cc[:,1]
    hpx_idx = hp.pixelfunc.ang2pix(NSIDE,lon,lat,lonlat=True)

th_list = np.logspace(np.log10(theta_min), np.log10(theta_max), num=nbins)
print('*** Theta values (deg):')
print(th_list)
#cl_list = [np.insert(ll,0,range(l_start))]
#cl_list = [idx]
#CCF_list = [th_list]
cl_list = np.zeros((ncl,N_stop-N_start+2))
cl_list[:,0] = np.insert(ll,0,range(l_start))[idx]
CCF_list = np.zeros((nbins,N_stop-N_start+2))
CCF_list[:,0] = th_list
A = (2.*ll+1.)/(4.*np.pi)
pl = []
for i in range(len(th_list)):
    pl.append(leg(ll,np.cos(np.radians(th_list[i]))))

print('*** factor for the correlation function:',fact)

if tag is not '':
    text = '#TAG: '+tag+'\n'+'#Maps number: '+str(N_start)+' - '+str(N_stop)+'\n'
    text_cl = '#TAG: '+tag+'\n'+'#Maps number: '+str(N_start)+' - '+str(N_stop)+'\n'
    text_CCF = '#TAG: '+tag+'\n'+'#Maps number: '+str(N_start)+' - '+str(N_stop)+'\n'
    tag = tag+'_'
else:
    text = '#Maps number: '+str(N_start)+' - '+str(N_stop)+'\n'
    text_cl = '#Maps number: '+str(N_start)+' - '+str(N_stop)+'\n'
    text_CCF = '#Maps number: '+str(N_start)+' - '+str(N_stop)+'\n'

text = text +'#N, N1h, N2h, alpha'
if add_noise:
    text = text + ', Cl_'+str(pivot_noise)+', N_norm, N_val'
if add_gauss_beam:
    text = text + ', beam_lev, beam_val (deg)'
if add_beam:
    text = text + ', beam decl'
text = text + '\n'

if out_dir is not '' and out_dir[-1] is not '/':
    out_dir = out_dir+'/'
out_text = out_dir+'msim_'+tag+str(N_start)+'_'+str(N_stop)+'.txt'
out_text_temp = out_dir+'msim_'+tag+'back.txt'
out_cl = out_dir+'Cl_'+tag+str(N_start)+'_'+str(N_stop)+'_label.dat'
out_CCF = out_dir+'CCF_'+tag+str(N_start)+'_'+str(N_stop)+'_label.dat'
if os.path.isfile(out_text):
    print('*** Parameters file already present!')
    f = open(out_text,'r')
    text = f.read()
    #print(text)
    cl_list = np.genfromtxt(out_cl)
    CCF_list = np.genfromtxt(out_CCF)
    #displace = len(cl_list[0])-1
    displace = len(np.where(cl_list[0]>0.)[0])-1
    print('*** Restarting from:')
    print(N_start+displace)
    #cl_list = list(np.transpose(cl_list))
    #print(cl_list)
    #CCF_list = list(np.transpose(CCF_list))
    #print(CCF_list)
    #exit()
else:
    displace = 0

def read_PS(path,in_1halo,in_2halo,ll,norm):
    cl1 = np.genfromtxt(path+in_1halo)
    cl1_interp = interp1d(np.log(cl1[:,0]),np.log(cl1[:,1]),fill_value='extrapolate')
    #print(cl1[:,0])
    cl1_fine = np.exp(cl1_interp(np.log(ll)))*norm
    cl2 = np.genfromtxt(path+in_2halo)
    cl2_interp = interp1d(np.log(cl2[:,0]),np.log(cl2[:,1]))
    #print(cl2[:,0])
    #exit()
    cl2_fine = np.exp(cl2_interp(np.log(ll)))*norm
    cl_tot = cl1_fine+cl2_fine
    cl_tot = np.insert(cl_tot,0,np.zeros(l_start))
    return cl_tot,np.insert(cl1_fine,0,np.zeros(l_start)),np.insert(cl2_fine,0,np.zeros(l_start))
cl_tot,cl_1h,cl_2h = read_PS(path,in_1halo,in_2halo,ll,norm)

t_start = time.time()

for i in range(N_start+displace,N_stop+1):
    cl_trans = []
    col = i - N_start+1
    #f = open(out_text_temp,'w')
    f = open(out_text,'w')
    print('*** Map number:',i)

    if verbose:
        print('Creating Power Spectrum...')
    N1h = np.round(10**random.uniform(np.log10(N1h_low),np.log10(N1h_up)),1)
    N2h = np.round(10**random.uniform(np.log10(N2h_low),np.log10(N2h_up)),1)
    alpha = np.round(random.uniform(alpha_low,alpha_up),1)
    #text = text.join(['{:}'.format(i),', ','{:.1e}'.format(N1h),', ','{:.1e}'.format(N2h),', ','{:.1e}'.format(alpha)])
    text = ''.join([text,'{:}'.format(i),', ','{:.1e}'.format(N1h),', ','{:.1e}'.format(N2h),', ','{:.1e}'.format(alpha)])
    if verbose:
        print('normalization 1-halo:',N1h)
        print('normalization 2-halo:',N2h)
        print('power-law index:',alpha)
    cl_1h_temp = cl_1h*N1h
    cl_2h_temp = cl_2h*N2h
    #cl_1h_temp[l_start:] = cl_1h_temp[l_start:]#*(ll/100.)**alpha
    cl_2h_temp[l_start:] = cl_2h_temp[l_start:]*(ll/100.)**alpha
    cl_temp = cl_1h_temp+cl_2h_temp
    #cl_list.append(cl_temp[idx])
    cl_list[:,col] = cl_temp[idx]
    if plot_test:
        plt.plot(range(len(cl_1h)),cl_1h,'--',linewidth=2,color='orange')
        plt.plot(range(len(cl_1h)),cl_2h,'.-',linewidth=2,color='orange')
        plt.plot(range(len(cl_1h)),cl_1h_temp,'--',linewidth=2,color='blue')
        plt.plot(range(len(cl_1h)),cl_2h_temp,'.-',linewidth=2,color='blue')
        plt.plot(idx,cl_1h_temp[idx]+cl_2h_temp[idx],'-',linewidth=2,color='black')
        plt.xlim(xmin=40,xmax=5000)
        plt.ylim(ymin=1.e-12,ymax=1.e-5)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
        #plt.savefig('test_PS.png')
        plt.clf()

    if verbose:
        print('Converting Power Spectrum to CCF...')
    for j in range(len(th_list)):
        cl_trans.append(np.sum(cl_temp[l_start:]*pl[j]*A))
    #CCF_list.append(np.array(cl_trans)*fact)
    CCF_list[:,col] = np.array(cl_trans)*fact

    if plot_test:
        plt.plot(th_list,CCF_list[:,col],'o-',linewidth=1,color='steelblue')
        plt.xlim(xmin=theta_min,xmax=theta_max)
        #plt.ylim(ymin=,ymax=)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
        #plt.savefig('test_CCF.png')
        plt.clf()

    out_name = out_dir+'msim_'+tag+str(i).zfill(5)+'.fits'
    if not add_noise:
        if verbose:
            print('Creating clean map from Power Spectrum...')
        msim = hp.synfast(cl_temp,NSIDE)
        if show_map:
            hp.mollview(msim)
            plt.show()
            plt.clf()
        if not reject_map:
            if verbose:
                print('Saving clean map...')
            hp.write_map(out_name,msim,coord='G',fits_IDL=False,overwrite=True)

    #moll_array = hp.cartview(msim, title=None, xsize=x_size, ysize=y_size, return_projected_map=True)
    #plt.savefig('/home/simone/RadioML/data/test/map_clean.png')
    if add_noise:
        #print('Creating noise from random normalization...')
        N = np.round(10**random.uniform(np.log10(N_low),np.log10(N_up)),1)
        if verbose:
            print('Noise normalization:',N)
            print('Noise pivot multipole:',pivot_noise)
        NN = cl_tot[pivot_noise]*N
        if verbose:
            print('Noise value:',NN)
        out_name = out_name+'_N_'+str(N)
        #text = text.join([', ','{:.3e}'.format(cl_tot[pivot_noise]),', ',str(N),', ','{:.3e}'.format(NN)])
        text = ''.join([text,', ','{:.3e}'.format(cl_tot[pivot_noise]),', ',str(N),', ','{:.3e}'.format(NN)])
        if plot_test:
            plt.plot(range(len(cl_1h)),cl_1h,'--',linewidth=2,color='orange')
            plt.plot(range(len(cl_1h)),cl_2h,'.-',linewidth=2,color='orange')
            plt.plot(range(len(cl_1h)),cl_1h_temp,'--',linewidth=2,color='blue')
            plt.plot(range(len(cl_1h)),cl_2h_temp,'.-',linewidth=2,color='blue')
            plt.plot(range(len(cl_1h)),cl_temp,'--',linewidth=2,color='black')
            plt.plot(range(len(cl_1h)),[NN]*len(cl_1h),'--',linewidth=2,color='black')
            plt.plot(range(len(cl_1h)),cl_temp+NN,'-',linewidth=2,color='black')
            plt.xlim(xmin=40,xmax=5000)
            plt.ylim(ymin=1.e-12,ymax=1.e-5)
            plt.xscale('log')
            plt.yscale('log')
            plt.show()
            #plt.savefig('test_PS.png')
            plt.clf()

        if verbose:
            print('Creating map from Power Spectrum with noise...')
        msim = hp.synfast(cl_temp+NN,NSIDE,new=True,lmax=l_stop)
        if not reject_map:
            hp.write_map(out_name,msim,coord='G',fits_IDL=False,overwrite=True)
        #    print('Combining maps...')
        #    msim = msim + mnoise
        if save_noise:
            out_noise = 'noise/noise_'+tag+str(i).zfill(5)+'_l'+str(pivot_noise)+'_N_'+str(N)+'.fits'
            if verbose:
                print('Creating noise map...')
            mnoise = hp.synfast([NN]*len(ll),NSIDE)
            if verbose:
                print('Saving noise map...')
            #print(path+out_noise)
            hp.write_map(path+out_noise,mnoise,coord='G',fits_IDL=False)

    if add_gauss_beam:
        b=np.round(10**random.uniform(np.log10(b_low),np.log10(b_up)),1)
        if verbose:
            print('Beam level:',b)
        out_name = out_name+'_b_'+str(b)
        if verbose:
            print('Convolving with beam...')
        pix_area = 4*np.pi/NPIX
        ang = np.sqrt(pix_area)*b
        if verbose:
            print('sigma:',np.degrees(ang))
        msim = hp.sphtfunc.smoothing(msim,sigma=ang)
        #text= text+', '+str(b)+', '+str(np.round(np.degrees(ang),5))
        text= ''.join([text,', '+str(b)+', '+str(np.round(np.degrees(ang),5))])

    if verbose:
        print('Converting map to cartesian projection...')
    if fast_patch:
        moll_array = msim[hpx_idx]
        moll_array = np.reshape(moll_array,(patch_GAL,patch_GAL))
    else:
        fig = plt.figure(1)
        moll_array = hp.cartview(msim,fig=1, title=None, xsize=x_size, ysize=y_size, return_projected_map=True)
        del msim

        if show_map:
            plt.show()
            plt.clf()

        #plt.savefig('/home/simone/RadioML/data/test/map_noise.png')
        if patch_GAL is not None:
            x_start = np.random.randint(0,high=(x_size-patch_GAL))
            y_del = (y_size-patch_GAL)//2
            moll_array = np.delete(moll_array,np.arange(y_size-y_del,y_size,dtype=int),axis=0)
            moll_array = np.delete(moll_array,np.arange(0,y_del,dtype=int),axis=0)
            moll_array = moll_array[:,x_start:x_start+patch_GAL]
            if verbose:
                print('Shape patch:')
                print(np.shape(moll_array))
            if show_map:
                plt.imshow(moll_array,vmin=-1.0,vmax=1.0)
                plt.colorbar()
                #plt.savefig('test_patch_GAL.pdf')
                plt.show()
                plt.clf()
                '''
                plt.imshow(map_patch,vmin=-1.0,vmax=1.0)
                plt.colorbar()
                plt.savefig('test_patch_GAL_new.pdf')
                #plt.show()
                plt.clf()
                plt.imshow(map_patch-moll_array,vmin=-1.0,vmax=1.0)
                plt.colorbar()
                plt.savefig('test_patch_GAL_diff.pdf')
                #plt.show()
                plt.clf()
                '''
    if not reject_clean:
        if not add_noise:
            out_tif = out_dir+'msim_'+tag+str(i).zfill(5)+'_clean.tif'
        else:
            out_tif = out_dir+'msim_'+tag+str(i).zfill(5)+'_wN.tif'
        moll_image = Image.fromarray(moll_array)
        moll_image.save(out_tif)
        del moll_image
    if add_beam:
        # Using pre-loaded beams
        if random_beam:
            k = np.random.randint(0,high=len(beams))
            if verbose:
                print('Convolving with beam:',beam_ids[k])
            ## ALTERNATIVE CONVOLUTION
            #from scipy import signal
            #moll_array_conv = signal.convolve2d(moll_array, beams[k], boundary='fill', mode='same')
            #
            from astropy.convolution import convolve
            moll_array_conv = convolve(moll_array, beams[k], normalize_kernel = False)

            if show_map:
                plt.imshow(moll_array_conv,vmin=-1.0,vmax=1.0)
                plt.colorbar()
                #plt.savefig('test_convolved_'+str(k)+'.pdf')
                plt.show()
                plt.clf()
            if norm_tif:
                if verbose:
                    print('Applying normalization to map...')
                moll_array_conv = normalization(moll_array_conv)
            moll_image_conv = Image.fromarray(moll_array_conv)
            # save it with tag for inclination in it
            if verbose:
                print('Saving tif map...')
            declination = beam_ids[k][len(beam_path)+47:-len('.fits')]
            out_tif = out_dir+'msim_'+tag+str(i).zfill(5)+'_'+str(declination).zfill(2)+'_data.tif'
            moll_image_conv.save(out_tif)
            del moll_array_conv
            del moll_image_conv

        else:
            if verbose:
                print('Convolving with beams...')
            for k, beam in enumerate(beams):
                ## ALTERNATIVE CONVOLUTION
                #time_c1 = time.time()
                #from scipy import signal
                #moll_array_conv = signal.convolve2d(moll_array, beams[k], boundary='fill', mode='same')
                #time_c11 = time.time()
                #print('Elapsed conv time 1:',time_c11 - time_c1)
                #if show_map:
                #    plt.imshow(moll_array_conv,vmin=-1.0,vmax=1.0)
                #    plt.colorbar()
                #    #plt.savefig('test_convolved_'+str(k)+'.pdf')
                #    plt.show()
                #    plt.clf()
                #

                #time_c2 = time.time()
                from astropy.convolution import convolve
                moll_array_conv = convolve(moll_array, beams[k], normalize_kernel = False)
                #time_c22 = time.time()
                #print('Elapsed conv time 2:',time_c22 - time_c2)

                if show_map:
                    plt.imshow(moll_array_conv,vmin=-1.0,vmax=1.0)
                    plt.colorbar()
                    #plt.savefig('test_convolved_'+str(k)+'.pdf')
                    plt.show()
                    plt.clf()
                if norm_tif:
                    if verbose:
                        print('Applying normalization to map...')
                    moll_array_conv = normalization(moll_array_conv)
                moll_image_conv = Image.fromarray(moll_array_conv)
                # save it with tag for inclination in it
                print('Saving tif map...')
                declination = beam_ids[k][len(beam_path)+47:-len('.fits')]
                out_tif = out_dir+'msim_'+tag+str(i).zfill(5)+'_'+str(declination).zfill(2)+'_data.tif'
                moll_image_conv.save(out_tif)
                del moll_array_conv
                del moll_image_conv

    else: #the map is saved once
        if norm_tif:
            if verbose:
                print('Applying normalization to map...')
            moll_array = normalization(moll_array)
            #moll_array = np.array(moll_array, dtype=np.uint8)
        if verbose:
            print('Converting map to tif format...')
        moll_image = Image.fromarray(moll_array)

        if verbose:
            print('Saving tif map...')
        out_tif = out_dir+'msim_'+tag+str(i).zfill(5)+'_data.tif'
        moll_image.save(out_tif)
        del moll_image

    del moll_array
    #plt.close('all')

    #text = text+'\n'
    text = ''.join([text,'\n'])
    if verbose:
        print('Writing parameters file...')
    f.write(text)
    f.close()
    #sub.call(['cp',out_text_temp,out_text],) #shell=[bool])
    if verbose:
        print('Writing Power Spectrum...')
    #np.savetxt(out_cl, np.transpose(cl_list), header=text_cl, fmt='%1.4e')
    np.savetxt(out_cl, cl_list, header=text_cl, fmt='%1.4e')
    if verbose:
        print('Writing CCF...')
    #np.savetxt(out_CCF, np.transpose(CCF_list), header=text_CCF, fmt='%1.4e')
    np.savetxt(out_CCF, CCF_list, header=text_CCF, fmt='%1.4e')

    if verbose:
        print('Partial time :',time.time()-t_start,'s')
    print('\n')
    #exit()

t_stop=time.time()
print('Elapsed time for create maps:',t_stop-t_start,'s')

print('Done.')

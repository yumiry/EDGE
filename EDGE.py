#!/usr/bin/env python
# Created by Dan Feldman and Connor Robinson for analyzing data from Espaillat Group research models.
# Last updated: 7/10/15 by Dan

#-------------------------------------------IMPORT RELEVANT MODELS-------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
#from astropy.io import ascii
from astropy.io import fits
#from matplotlib.backends.backend_pdf import PdfPages
import os
import math
import cPickle
import pdb

#----------------------------------------------PLOTTING PARAMETERS-----------------------------------------------
# Regularizes the plotting parameters like tick sizes, legends, etc.
plt.rc('xtick', labelsize='medium')
plt.rc('ytick', labelsize='medium')
plt.rc('text', usetex=True)
plt.rc('legend', fontsize=10)
plt.rc('axes', labelsize=15)
plt.rc('figure', autolayout=True)

#-----------------------------------------------------PATHS------------------------------------------------------
# Folders where model output data and observational data can be found:
datapath        = '/Users/danfeldman/Orion_Research/Orion_Research/CVSO_4Objs/Models/CVSO109PT2/'
#figurepath      = '/Users/danfeldman/Orion_Research/Orion_Research/CVSO_4Objs/Look_SEDs/CVSO90PT/'
figurepath      = '/Users/danfeldman/Orion_Research/Orion_Research/CVSO_4Objs/Models/CVSO107_2/'
#deredpath       = '/Users/danfeldman/Orion_Research/Dereddening_Codes/starparam/'           # De-redden magnitude path

#---------------------------------------------INDEPENDENT FUNCTIONS----------------------------------------------
# A function is considered independent if it does not reference any other function or class in this module.
def filelist(path):
    """
    (By Dan)
    Returns a list of files in a directory. Pops out hidden values.
    
    INPUTS
    path: The directory path from which we wish to grab a file list.
    
    OUTPUT
    flist: The file list. It should be devoid of hidden files.
    """
    
    flist       = os.listdir(path)                              # Full list of files
    hid         = []
    for f in flist:
        if f.startswith('.'):                                   # If hidden file, tag it
            hid.append(flist.index(f))
    for index, val in enumerate(hid):                           # Pop out tagged entries
        flist.pop(val - index)
    return flist

def convertFreq(value):
    """
    (By Dan)
    Convert a frequency value in s-1 to wavelength in microns. Should also work with arrays.
    
    INPUTS
    value: A frequency value or array of frequency values in s-1 units.
    
    OUTPUT
    wl: The wavelength or array of wavelength values in microns.
    """
    
    c_microns   = 2.997924e14                                   # Speed of light in microns
    wl          = c_microns / value
    
    return wl

def convertJy(value, wavelength):
    """
    (By Dan)
    Convert a flux in Janskys to erg s-1 cm-2. Should also work with flux/wl arrays of same size.
    
    INPUTS
    value: A flux value in the units of Jy.
    wavelength: The corresponding wavelength value (or perhaps a central wavelength).
    
    OUTPUT
    flux: The flux value in units of erg s-1 cm-2.
    """
    
    c_microns   = 2.997924e14                                   # Speed of light in microns
    flux        = value * 1e-23 * (c_microns / wavelength)      # lamda*F_lambda or nu*F_nu
    
    return flux

def convertMag(value, band, jy='False'):
    """
    (By Dan)
    Converts a magnitude into a flux in erg s-1 cm-2. To use this for an array, use np.vectorize().
    Currently handles:
        UBVRI
        JHK
        LMNQ 
        griz
        MIPS(24,70,160)
        IRAC (3.6,4.5,5.8,8.0)
        W1-W4 (WISE)
        
    References: http://people.physics.tamu.edu/lmacri/astro603/lectures/astro603_lect01.pdf
                http://casa.colorado.edu/~ginsbura/filtersets.htm
                http://www.astro.utoronto.ca/~patton/astro/mags.html
                http://ircamera.as.arizona.edu/astr_250/Lectures/Lecture_13.htm
                
    INPUTS
    value: A magnitude value (units of mag).
    band: The band corresponding to the magnitude value.
    jy: Boolean -- If False, will use convertJy() to convert the flux into erg s-1 cm-2. If True, will
                   leave the output value in Jy.
    
    OUTPUTS
    flux: The flux value in erg s-1 cm-2.
    fluxJ: The flux value in Jy.
    """
    
    # First convert to Janskys:
    if band.upper()     == 'U':
        fluxJ       = 1810. * (10.0**(value / -2.5))
        wavelength  = 0.367                                     # In Microns                               
    elif band.upper()   == 'B':
        fluxJ       = 4260. * (10.0**(value / -2.5))
        wavelength  = 0.436                                    
    elif band.upper()   == 'V':
        fluxJ       = 3640. * (10.0**(value / -2.5))
        wavelength  = 0.545                                     
    elif band.upper()   == 'R':
        fluxJ       = 3080. * (10.0**(value / -2.5))
        wavelength  = 0.638
    elif band.upper()   == 'I':
        fluxJ       = 2550. * (10.0**(value / -2.5))
        wavelength  = 0.797
    elif band.upper()   == 'J':
        fluxJ       = 1600. * (10.0**(value / -2.5))
        wavelength  = 1.220
    elif band.upper()   == 'H':
        fluxJ       = 1080. * (10.0**(value / -2.5))
        wavelength  = 1.630
    elif band.upper()   == 'K':
        fluxJ       = 670. * (10.0**(value / -2.5))
        wavelength  = 2.190
    elif band.upper()   == 'L':
        fluxJ       = 281. * (10.0**(value / -2.5))
        wavelength  = 3.450
    elif band.upper()   == 'M':
        fluxJ       = 154. * (10.0**(value / -2.5))
        wavelength  = 4.750
    elif band.upper()   == 'N':
        fluxJ       = 37. * (10.0**(value / -2.5))
        wavelength  = 10.10
    elif band.upper()   == 'Q':
        fluxJ       = 10. * (10.0**(value / -2.5))
        wavelength  = 20.00
    elif band.upper()   == 'SDSSG':
        fluxJ       = 3730. * (10.0**(value / -2.5))
        wavelength  = 0.4686
    elif band.upper()   == 'SDSSR':
        fluxJ       = 4490. * (10.0**(value / -2.5))
        wavelength  = 0.6165
    elif band.upper()   == 'SDSSI':
        fluxJ       = 4760. * (10.0**(value / -2.5))
        wavelength  = 0.7481
    elif band.upper()   == 'SDSSZ':
        fluxJ       = 4810. * (10.0**(value / -2.5))
        wavelength  = 0.8931
    elif band.upper()   == 'MIPS24':
        fluxJ       = 7.17 * (10.0**(value / -2.5))
        wavelength  = 23.675
    elif band.upper()   == 'MIPS70':
        fluxJ       = 0.778 * (10.0**(value / -2.5))
        wavelength  = 71.42
    elif band.upper()   == 'MIPS160':
        fluxJ       = 0.16 * (10.0**(value / -2.5))
        wavelength  = 155.9
    elif band.upper()   == 'IRAC3.6':
        fluxJ       = 280.9 * (10.0**(value / -2.5))
        wavelength  = 3.60
    elif band.upper()   == 'IRAC4.5':
        fluxJ       = 179.7 * (10.0**(value / -2.5))
        wavelength  = 4.50
    elif band.upper()   == 'IRAC5.8':
        fluxJ       = 115. * (10.0**(value / -2.5))
        wavelength  = 5.80
    elif band.upper()   == 'IRAC8.0':
        fluxJ       = 64.13 * (10.0**(value / -2.5))
        wavelength  = 8.0
    elif band.upper()   == 'W1':
        fluxJ       = 309.5 * (10.0**(value / -2.5))
        wavelength  = 3.35
    elif band.upper()   == 'W2':
        fluxJ       = 171.8 * (10.0**(value / -2.5))
        wavelength  = 4.60
    elif band.upper()   == 'W3':
        fluxJ       = 31.67 * (10.0**(value / -2.5))
        wavelength  = 11.56
    elif band.upper()   == 'W4':
        fluxJ       = 8.36 * (10.0**(value / -2.5))
        wavelength  = 22.09
    else:
        raise ValueError('CONVERTOPTMAG: Unknown Band given. Cannot convert.')
    
    if jy == 'False':
        # Next, convert to flux from Janskys:
        flux        = convertJy(fluxJ, wavelength)              # Ok, so maybe this is a dependent function...
        return flux                                             # Shhhhhhh! :)
    return fluxJ

def numCheck(num, high=0):
    """
    (By Dan)
    Takes a number between 0 and 9999 and converts it into a 3 or 4 digit string. E.g., 2 --> '002', 12 --> '012'
    
    INPUT
    num: A number between 0 and 9999. If this is a float, it will still work, but it will chop off the decimal.
    high: BOOLEAN -- if True (1), output is forced to be a 4 digit string regardless of the number.
        
    OUTPUT
    numstr: A string of 3 or 4 digits, where leading zeroes fill in any spaces.
    
    """
    if num > 9999 or num < 0:
        raise ValueError('Number too small/large for string handling!')
    if num > 999 or high == 1: 
        numstr          = '%04d' % num
    else:
        numstr          = '%03d' % num
    return numstr

#----------------------------------------------DEPENDENT FUNCTIONS-----------------------------------------------
# A function is considered dependent if it utilizes either the above independent functions, or the classes below.
def look(obs, model=None, jobn=None, save=0, colkeys=None, diskcomb=0):
    """
    (By Dan)
    Creates a plot of a model and the observations for a given target.
    
    INPUTS
    model: The object containing the target's model. Should be an instance of the TTS_Model class. This is an optional input.
    obs: The object containing the target's observations. Should be an instance of the TTS_Obs class.
    jobn: The "job number." This is meaningless for observation-only plots, but if you save the file, we require a number.
    save: BOOLEAN -- If 1 (True), will save the plot in a pdf file. If 0 (False), will output to screen.
    colkeys: An optional input array of color strings. This can be used to overwrite the normal color order convention. Options include:
             p == purple, r == red, m == magenta, b == blue, c == cyan, l == lime, t == teal, g == green, y == yellow, o == orange,
             k == black, w == brown, v == violet, d == gold, n == pumpkin, e == grape, j == jeans, s == salmon
             If not specified, the default order will be used, and once you run out, we'll have an error. So if you have more than 18
             data types, you'll need to supply the order you wish to use (and which to repeat). Or you can add new colors using html tags
             to the code, and then update this header.
    diskcomb: BOOLEAN -- If 1 (True), will combine outer wall and disk components into one for plotting. If 0 (False), will separate.
    
    OUTPUT
    A plot. Can be saved or plotted to the screen based on the "save" input parameter.
    """

    photkeys            = obs.photometry.keys()         # obs.photometry and obs.spectra are dictionaries.
    speckeys            = obs.spectra.keys()
    colors              = {'p':'#7741C8', 'r':'#F50C0C', 'm':'#F50CA3', 'b':'#2B0CF5', 'c':'#0CE5F5', 'l':'#33F50C', 't':'#4DCE9B', \
                           'g':'#1D5911', 'y':'#BFB91E', 'o':'#F2A52A', 'k':'#060605', 'w':'#5A3A06', 'v':'#BD93D2', 'd':'#FFD900', \
                           'n':'#FF7300', 'e':'#9A00FA', 'j':'#00AAFF', 's':'#D18787'}
    if colkeys == None:
        colkeys         = ['p', 'r', 'o', 'b', 'c', 'm', 'g', 'y', 'l', 'k', 't', 'w', 'v', 'd', 'n', 'e', 'j', 's']    # Order in which colors are used

    # Let the plotting begin!
    plt.figure(1)#,figsize=(3,4))
    # Plot the spectra first:
    for sind, skey in enumerate(speckeys):
        plt.plot(obs.spectra[skey]['wl'], obs.spectra[skey]['lFl'], color=colors[colkeys[sind]] , linewidth=2.0, label=skey)
    # Next is the photometry:
    for pind, pkey in enumerate(photkeys):
        # If an upper limit only:
        if pkey in obs.ulim:
            #plt.arrow(obs.photometry[pkey]['wl'], obs.photometry[pkey]['lFl'], 0.0, -1.*(obs.photometry[pkey]['lFl']/2.), \
            #          color=colors[colkeys[pind+len(speckeys)]], length_includes_head=False)
            plt.plot(obs.photometry[pkey]['wl'], obs.photometry[pkey]['lFl'], 'v', \
                     color=colors[colkeys[pind+len(speckeys)]], markersize=7, label=pkey)
        # If not an upper limit, plot as normal:
        else:
            if 'err' not in obs.photometry[pkey].keys():
                plt.plot(obs.photometry[pkey]['wl'], obs.photometry[pkey]['lFl'], 'o', mfc='w', mec=colors[colkeys[pind+len(speckeys)]], mew=1.0,\
                         markersize=7, label=pkey)
            else:
                plt.errorbar(obs.photometry[pkey]['wl'], obs.photometry[pkey]['lFl'], yerr=obs.photometry[pkey]['err'], mec=colors[colkeys[pind+len(speckeys)]], \
                             fmt='o', mfc='w', mew=1.0, markersize=7, ecolor=colors[colkeys[pind+len(speckeys)]], elinewidth=2.0, capsize=2.0, label=pkey)
    # Now, the model (if a model supplied):
    if model != None:
        modkeys         = model.data.keys()
        if 'phot' in modkeys:
            plt.plot(model.data['wl'], model.data['phot'], ls='--', c='b', linewidth=2.0, label='Photosphere')
        if 'iwall' in modkeys:
            plt.plot(model.data['wl'], model.data['iwall'], ls='--', c='#53EB3B', linewidth=2.0, label='Inner Wall')
        if diskcomb:
            try:
                diskflux    = model.data['owall'] + model.data['disk']
            except KeyError:
                print 'LOOK: Error, tried to combine outer wall and disk components but one component is missing!'
            else:    
                plt.plot(model.data['wl'], diskflux, ls='--', c='#8B0A1E', linewidth=2.0, label='Outer Disk')
        else:
            if 'owall' in modkeys:
                plt.plot(model.data['wl'], model.data['owall'], ls='--', c='#E9B021', linewidth=2.0, label='Outer Wall')
            if 'disk' in modkeys:
                plt.plot(model.data['wl'], model.data['disk'], ls='--', c='#8B0A1E', linewidth=2.0, label='Disk')
        if 'dust' in modkeys:
            plt.plot(model.data['wl'], model.data['dust'], ls='--', c='#F80303', linewidth=2.0, label='Opt. Thin Dust')
        if 'total' in modkeys:
            plt.plot(model.data['wl'], model.data['total'], c='k', linewidth=2.0, label='Combined Model')
        # Now, the relevant meta-data:
        plt.figtext(0.75,0.88,'Eps = '+ str(model.eps), color='#E5BF03', size='9')
        plt.figtext(0.75,0.85,'Alpha = '+ str(model.alpha), color='#FA9F00', size='9')
        plt.figtext(0.75,0.82,'Amax = '+ str(model.amax), color='#D78A04', size='9')
        plt.figtext(0.75,0.79,'Rin = '+ str(model.rin), color='#C47D03', size='9')
        plt.figtext(0.75,0.76,'Rout = '+ str(model.rdisk), color='#9A6202', size='9')
        plt.figtext(0.75,0.73,'IWallT = '+ str(model.temp), color='#815201', size='9')
        plt.figtext(0.75,0.70,'Altinh = '+ str(model.altinh), color='#5B3A00', size='9')
        plt.figtext(0.75,0.67,'Mdot = '+ str(model.mdot), color='#3D2C02', size='9')
    # Lastly, the remaining parameters to plotting (mostly aesthetics):
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(2e-1, 2e3)
    plt.ylim(1e-15, 1e-9)
    plt.ylabel(r'${\rm \lambda F_{\lambda}\; (erg\; s^{-1}\; cm^{-2})}$')
    plt.xlabel(r'${\rm {\bf \lambda}\; (\mu m)}$')
    plt.title(obs.name.upper())
    plt.legend(loc=3)
    if save:
        if type(jobn) != int:
            raise ValueError('LOOK: Jobn must be an integer if you wish to save the plot.')
        jobstr          = numCheck(jobn)
        plt.savefig(figurepath + obs.name.upper() + '_' + jobstr + '.pdf', dpi=350)
    else:
        plt.show()
    plt.clf()

    return

def searchJobs(target, dpath=datapath, **kwargs):
    """
    (By Dan)
    Searches through the job file outputs to determine which jobs (if any) matches the set of input parameters.
    
    INPUTS
    target: The name of the target we're checking against (e.g., cvso109, DMTau, etc.).
    **kwargs: Any keyword arguments (kwargs) supplied. These should correspond to the header filenames (not case sensitive). The code
              will loop through each of these kwargs and see if they all match.
    
    OUTPUTS
    job_matches: A numpy array containing all the jobs that matched the kwargs. Can be an empty array, single value array, or multivalued array. Will
                 contain matches by their integer number.
    
    """
    
    job_matches         = np.array([], dtype='string')
    targList            = filelist(dpath)
    
    # Pop out all files that do not correspond to jobs:
    not_data            = []
    for f in targList:
        if f.startswith(target) and f.endswith('.fits'):
            continue
        else:
            not_data.append(targList.index(f))
    for ind, val in enumerate(not_data):
        targList.pop(val - ind)
    
    # Now go through the list and find any jobs matching the desired input parameters:
    for jobi, job in enumerate(targList):
        if 'OTD' in job:
            continue
        fitsF           = fits.open(dpath+job)
        header          = fitsF[0].header
        for kwarg, value in kwargs.items():
            if header[kwarg.upper()] != value:
                break
        else:
            # Check if three or four string number:
            if job[-9] == '_':
                job_matches = np.append(job_matches, job[-8:-5])
            else:
                job_matches = np.append(job_matches, job[-9:-5])
        fitsF.close()
    
    return job_matches

def loadPickle(name, picklepath=datapath, num=None):
    """
    (By Dan)
    Loads in a pickle saved from the TTS_Obs class.
    
    INPUTS
    name: The name of the object whose observations are stored in the pickle.
    picklepath: The directory location of pickle. Default path is datapath, defined at top of this module.
    num: An optional number provided if there are multiple pickles for this object and you want to load a specific one.
    
    OUTPUT
    pickle: The object containing the data loaded in from the pickle.
    
    """
    if num == None:
        # Check if there is more than one
        flist           = filelist(picklepath)
        if (name + '_obs_1.pkl') in flist:
            print 'LOADPICKLE: Warning! There is more than one pickle file for this object! Make sure it is the right one!'
        f               = open(picklepath+name+'_obs.pkl', 'rb')
        pickle          = cPickle.load(f)
        f.close()
    elif num != None:
        f               = open(picklepath+name+'_obs_'+str(num)+'.pkl', 'rb')
        pickle          = cPickle.load(f)
        f.close()
    return pickle

def job_file_create(jobnum, path, high=0, **kwargs):
    """
    (By Dan)
    Creates a new job file that is used by the D'Alessio Model.
    
    INPUTS
    jobnum: The job number used to name the output job file.
    path: The path containing the sample job file, and ultimately, the output.
    high: BOOLEAN -- if True (1), output will be jobXXXX instead of jobXXX.
    **kwargs: The keywords arguments used to make changes to the sample file. Available
              kwargs include:
        amaxs - maximum grain size in disk
        epsilon - settling parameter
        mstar - mass of protostar
        tstar - effective temperature of protostar
        rstar - radius of protostar
        dist - distance to the protostar (or likely, the cluster it's in)
        mdot - the mass accretion rate of protostellar system
        tshock - the temperature of the shock
        alpha - the alpha viscosity parameter
        mui - the cosine of the inclination angle
        rdisk - the outer radius of the disk
        labelend - the labelend of all output files when job file is run
        temp - the temperature of the inner wall
        altinh - the height of the inner wall in scale heights
        
        Some can still be included, such as dust grain compositions. They just aren't
        currently supported.
    
    OUTPUT
    A job file with the name jobXXX, where XXX is the three-string number from 001 - 999. If
    high == True, the output name will be jobXXXX, where XXXX is a four-string number from 1000-9999.
    No formal outputs are returned by this function; the file is created in the path directory.
    
    """
    
    # First, let's read in the sample job file so we have a template:
    job_file = open(path+'job_sample', 'r')
    fullText = job_file.readlines()     # All text in a list of strings
    job_file.close()
    
    # Double check for the correct default amax and epsilon values:
    if fullText[43][0] == '#' or fullText[44][0] == '#':
        raise ValueError('JOB_FILE_CREATE: There is a comment problem at amax=0.25')
    if fullText[71][0] == '#' or fullText[72][0] == '#':
        raise ValueError('JOB_FILE_CREATE: There is a comment problem at eps=0.0001')
    
    # Now we run through the list of changes desired and change them:
    # If we want to change the maximum grain size (amaxs):
    if 'amaxs' in kwargs:
        # amaxs is a commented out switch, so we need to know the desired size:
        if kwargs['amaxs'] == 0.25:
            pass
        elif kwargs['amaxs'] == 0.05:
            if fullText[37][0] == '#' and fullText[38][0] == '#':
                fullText[37] = fullText[37][1:]     # Remove the pound at 0.05
                fullText[38] = fullText[38][1:]
                fullText[43] = '#' + fullText[43]   # Add the pound at 0.25
                fullText[44] = '#' + fullText[44]
            else:
                raise ValueError('JOB_FILE_CREATE: There is a comment problem at amax=0.05')
        elif kwargs['amaxs'] == 0.1:
            if fullText[40][0] == '#' and fullText[41][0] == '#':
                fullText[40] = fullText[40][1:]     # Remove the pound at 0.1
                fullText[41] = fullText[41][1:]
                fullText[43] = '#' + fullText[43]   # Add the pound at 0.25
                fullText[44] = '#' + fullText[44]
            else:
                raise ValueError('JOB_FILE_CREATE: There is a comment problem at amax=0.1')
        elif kwargs['amaxs'] == 1.0:
            if fullText[46][0] == '#' and fullText[47][0] == '#':
                fullText[46] = fullText[46][1:]     # Remove the pound at 1.0
                fullText[47] = fullText[47][1:]
                fullText[43] = '#' + fullText[43]   # Add the pound at 0.25
                fullText[44] = '#' + fullText[44]
            else:
                raise ValueError('JOB_FILE_CREATE: There is a comment problem at amax=1.0')
        elif kwargs['amaxs'] == 2.0:
            if fullText[49][0] == '#' and fullText[50][0] == '#':
                fullText[49] = fullText[49][1:]     # Remove the pound at 2.0
                fullText[50] = fullText[50][1:]
                fullText[43] = '#' + fullText[43]   # Add the pound at 0.25
                fullText[44] = '#' + fullText[44]
            else:
                raise ValueError('JOB_FILE_CREATE: There is a comment problem at amax=2.0')
        elif kwargs['amaxs'] == 3.0:
            if fullText[52][0] == '#' and fullText[53][0] == '#':
                fullText[52] = fullText[52][1:]     # Remove the pound at 3.0
                fullText[53] = fullText[53][1:]
                fullText[43] = '#' + fullText[43]   # Add the pound at 0.25
                fullText[44] = '#' + fullText[44]
            else:
                raise ValueError('JOB_FILE_CREATE: There is a comment problem at amax=3.0')
        elif kwargs['amaxs'] == 4.0:
            if fullText[55][0] == '#' and fullText[56][0] == '#':
                fullText[55] = fullText[55][1:]     # Remove the pound at 4.0
                fullText[56] = fullText[56][1:]
                fullText[43] = '#' + fullText[43]   # Add the pound at 0.25
                fullText[44] = '#' + fullText[44]
            else:
                raise ValueError('JOB_FILE_CREATE: There is a comment problem at amax=4.0')
        elif kwargs['amaxs'] == 5.0:
            if fullText[58][0] == '#' and fullText[59][0] == '#':
                fullText[58] = fullText[58][1:]     # Remove the pound at 5.0
                fullText[59] = fullText[59][1:]
                fullText[43] = '#' + fullText[43]   # Add the pound at 0.25
                fullText[44] = '#' + fullText[44]
            else:
                raise ValueError('JOB_FILE_CREATE: There is a comment problem at amax=5.0')
        elif kwargs['amaxs'] == 10.0:
            if fullText[61][0] == '#' and fullText[62][0] == '#':
                fullText[61] = fullText[61][1:]     # Remove the pound at 10.0
                fullText[62] = fullText[62][1:]
                fullText[43] = '#' + fullText[43]   # Add the pound at 0.25
                fullText[44] = '#' + fullText[44]
            else:
                raise ValueError('JOB_FILE_CREATE: There is a comment problem at amax=10.0')
        elif kwargs['amaxs'] == 100.0:
            if fullText[64][0] == '#' and fullText[65][0] == '#':
                fullText[64] = fullText[64][1:]     # Remove the pound at 100.0
                fullText[65] = fullText[65][1:]
                fullText[43] = '#' + fullText[43]   # Add the pound at 0.25
                fullText[44] = '#' + fullText[44]
            else:
                raise ValueError('JOB_FILE_CREATE: There is a comment problem at amax=100.0')
        else:
            raise ValueError('JOB_FILE_CREATE: Invalid input for AMAXS!')
    
    # Now, we examine the epsilon parameter if a value provided:
    if 'epsilon' in kwargs:
        # Epsilon is a commented out switch, so we need the desired parameter:
        if kwargs['epsilon'] == 0.0001:
            pass        # Default value is 0.0001
        elif kwargs['epsilon'] == 0.001:
            if fullText[74][0] == '#' and fullText[75][0] == '#':
                fullText[74] = fullText[74][1:]     # Remove the pound at 0.001
                fullText[75] = fullText[75][1:]
                fullText[71] = '#' + fullText[71]   # Add the pound at 0.0001
                fullText[72] = '#' + fullText[72]
            else:
                raise ValueError('JOB_FILE_CREATE: There is a comment problem at eps=0.001')
        elif kwargs['epsilon'] == 0.01:
            if fullText[77][0] == '#' and fullText[78][0] == '#':
                fullText[77] = fullText[77][1:]     # Remove the pound at 0.01
                fullText[78] = fullText[78][1:]
                fullText[71] = '#' + fullText[71]   # Add the pound at 0.0001
                fullText[72] = '#' + fullText[72]
            else:
                raise ValueError('JOB_FILE_CREATE: There is a comment problem at eps=0.01')
        elif kwargs['epsilon'] == 0.1:
            if fullText[80][0] == '#' and fullText[81][0] == '#':
                fullText[80] = fullText[80][1:]     # Remove the pound at 0.1
                fullText[81] = fullText[81][1:]
                fullText[71] = '#' + fullText[71]   # Add the pound at 0.0001
                fullText[72] = '#' + fullText[72]
            else:
                raise ValueError('JOB_FILE_CREATE: There is a comment problem at eps=0.1')
        elif kwargs['epsilon'] == 0.2:
            if fullText[83][0] == '#' and fullText[84][0] == '#':
                fullText[83] = fullText[83][1:]     # Remove the pound at 0.2
                fullText[84] = fullText[84][1:]
                fullText[71] = '#' + fullText[71]   # Add the pound at 0.0001
                fullText[72] = '#' + fullText[72]
            else:
                raise ValueError('JOB_FILE_CREATE: There is a comment problem at eps=0.2')
        elif kwargs['epsilon'] == 0.5:
            if fullText[86][0] == '#' and fullText[87][0] == '#':
                fullText[86] = fullText[86][1:]     # Remove the pound at 0.5
                fullText[87] = fullText[87][1:]
                fullText[71] = '#' + fullText[71]   # Add the pound at 0.0001
                fullText[72] = '#' + fullText[72]
            else:
                raise ValueError('JOB_FILE_CREATE: There is a comment problem at eps=0.5')
        elif kwargs['epsilon'] == 1.0:
            if fullText[89][0] == '#' and fullText[90][0] == '#':
                fullText[89] = fullText[89][1:]     # Remove the pound at 1.0
                fullText[90] = fullText[90][1:]
                fullText[71] = '#' + fullText[71]   # Add the pound at 0.0001
                fullText[72] = '#' + fullText[72]
            else:
                raise ValueError('JOB_FILE_CREATE: There is a comment problem at eps=1.0')
        else:
            raise ValueError('JOB_FILE_CREATE: Invalid input for epsilon!')
    
    # Now we can cycle through the easier changes desired:
    if 'mstar' in kwargs:                           # Stellar mass parameter
        fullText[14] = (fullText[14][:11] + str(kwargs['mstar']) + 
                        fullText[14][-11:])
    if 'tstar' in kwargs:                           # Photosphere temp parameter
        fullText[15] = (fullText[15][:11] + str(kwargs['tstar']) + 
                        fullText[15][-8:])
    if 'rstar' in kwargs:                           # Stellar radius parameter
        fullText[16] = (fullText[16][:11] + str(kwargs['rstar']) + 
                        fullText[16][-12:])
    if 'dist' in kwargs:                            # Stellar distance parameter
        fullText[17] = (fullText[17][:15] + str(kwargs['dist']) + 
                        fullText[17][-14:])
    if 'mdot' in kwargs:                            # Accretion rate parameter
        fullText[18] = (fullText[18][:10] + str(kwargs['mdot']) + 
                        fullText[18][-15:])
    if 'tshock' in kwargs:                          # Shock temp parameter
        fullText[21] = fullText[21][:11] + str(float(kwargs['tshock'])) + '\n'
    if 'alpha' in kwargs:                           # Alpha viscosity parameter
        fullText[24] = (fullText[24][:11] + str(kwargs['alpha']) + 
                        fullText[24][-2:])
    if 'mui' in kwargs:                             # Cosine of inclination
        fullText[25] = (fullText[25][:9] + str(kwargs['mui']) + 
                        fullText[25][-32:])
    if 'rdisk' in kwargs:                           # Outer disk radius parameter
        fullText[31] = (fullText[31][:11] + str(kwargs['rdisk']) + 
                        fullText[31][-25:])
    if 'labelend' in kwargs:                        # Labelend on output files
        fullText[159] = (fullText[159][:14] + str(kwargs['labelend']) + 
                        fullText[159][-2:])
    if 'temp' in kwargs:                            # Inner wall temp parameter
        fullText[562] = (fullText[562][:9] + str(kwargs['temp']) + 
                        fullText[562][-2:])
    if 'altinh' in kwargs:                          # Inner wall height parameter
        fullText[563] = (fullText[563][:11] + str(kwargs['altinh']) + 
                        fullText[563][-2:])
    
    # Once all changes have been made, we just create a new job file:
    if high:
        string_num = numCheck(jobnum, high=True)
    else:
        string_num  = numCheck(jobnum)
    newJob      = open(path+'job'+string_num, 'w')
    newJob.writelines(fullText)
    newJob.close()
    
    return
    
def job_optthin_create(jobn, path, high=0, **kwargs):
    """
    (By Dan)
    Creates a new optically thin dust job file.
    
    INPUTS
    jobn: The job number used to name the output job file.
    path: The path containing the sample job file, and ultimately, the output.
    high: BOOLEAN -- if True (1), output will be job_optthinXXXX instead of job_optthinXXX.
    **kwargs: The keywords arguments used to make changes to the sample file. Available
              kwargs include:
        amax - maximum grain size
        tstar - effective temperature of protostar
        rstar - radius of protostar
        dist - distance to the protostar (or likely, the cluster it's in)
        mui - the cosine of the inclination angle
        rdisk - the outer radius
        rin - the inner radius
        labelend - the labelend of all output files when job file is run
        tau - optical depth, I think
        power - no idea what this one is
        fudgeorg - don't know this one either
        fudgetroi - or this one...should probably look this up
        fracsil - fraction of silicates by mass
        fracent - fraction of enstatite by mass
        fracforst - fraction of forsterite by mass
        fracamc - fraction of amorphous carbon by mass
        
        Some can still be included, such as dust grain compositions. They just aren't
        currently supported.
    
    OUTPUT
    A job file with the name job_optthinXXX, where XXX is the three-string number from 001 - 999. If
    high == True, the output name will be job_optthinXXXX, where XXXX is a four-string number from 1000-9999.
    No formal outputs are returned by this function; the file is created in the path directory.
    """
    
    # First, load in the sample job file for a template:
    job_file = open(path+'job_optthin_sample', 'r')
    fullText = job_file.readlines()     # All text in a list of strings
    job_file.close()
    
    # Double check for the correct default amax value:
    if fullText[30][0] == '#':
        raise ValueError('JOB_OPTTHIN_CREATE: There is a comment problem at amax = 0.25!')
    
    # Now we run through the list of changes desired and change them:
    # If we want to change amax:
    if 'amax' in kwargs:
        # amax is a commented out switch, so we need to know the desired size:
        if kwargs['amax'] == 0.25:
            pass
        elif kwargs['amax'] == 0.05 or kwargs['amax'] == '0p05':
            if fullText[28][0] == '#':
                fullText[28] = fullText[28][1:]     # Remove the pound at 0.05
                fullText[30] = '#' + fullText[30]   # Add the pound at 0.25
            else:
                raise ValueError('JOB_OPTTHIN_CREATE: There is a comment problem at amax = 0.05!')
        elif kwargs['amax'] == 0.1 or kwargs['amax'] == '0p1':
            if fullText[29][0] == '#':
                fullText[29] = fullText[29][1:]     # Remove the pound at 0.1
                fullText[30] = '#' + fullText[30]   # Add the pound at 0.25
            else:
                raise ValueError('JOB_OPTTHIN_CREATE: There is a comment problem at amax = 0.1!')
        elif kwargs['amax'] == 1.0 or kwargs['amax'] == '1p0':
            if fullText[31][0] == '#':
                fullText[31] = fullText[31][1:]     # Remove the pound at 1.0
                fullText[30] = '#' + fullText[30]   # Add the pound at 0.25
            else:
                raise ValueError('JOB_OPTTHIN_CREATE: There is a comment problem at amax = 1.0!')
        elif kwargs['amax'] == 2.0 or kwargs['amax'] == '2p0':
            if fullText[32][0] == '#':
                fullText[32] = fullText[32][1:]     # Remove the pound at 2.0
                fullText[30] = '#' + fullText[30]   # Add the pound at 0.25
            else:
                raise ValueError('JOB_OPTTHIN_CREATE: There is a comment problem at amax = 2.0!')
        elif kwargs['amax'] == 3.0 or kwargs['amax'] == '3p0':
            if fullText[33][0] == '#':
                fullText[33] = fullText[33][1:]     # Remove the pound at 3.0
                fullText[30] = '#' + fullText[30]   # Add the pound at 0.25
            else:
                raise ValueError('JOB_OPTTHIN_CREATE: There is a comment problem at amax = 3.0!')
        elif kwargs['amax'] == 4.0 or kwargs['amax'] == '4p0':
            if fullText[34][0] == '#':
                fullText[34] = fullText[34][1:]     # Remove the pound at 4.0
                fullText[30] = '#' + fullText[30]   # Add the pound at 0.25
            else:
                raise ValueError('JOB_OPTTHIN_CREATE: There is a comment problem at amax = 4.0!')
        elif kwargs['amax'] == 5.0 or kwargs['amax'] == '5p0':
            if fullText[35][0] == '#':
                fullText[35] = fullText[35][1:]     # Remove the pound at 5.0
                fullText[30] = '#' + fullText[30]   # Add the pound at 0.25
            else:
                raise ValueError('JOB_OPTTHIN_CREATE: There is a comment problem at amax = 5.0!')
        elif kwargs['amax'] == 10.0 or kwargs['amax'] == '10':
            if fullText[36][0] == '#':
                fullText[36] = fullText[36][1:]     # Remove the pound at 10.0
                fullText[30] = '#' + fullText[30]   # Add the pound at 0.25
            else:
                raise ValueError('JOB_OPTTHIN_CREATE: There is a comment problem at amax = 10!')
        elif kwargs['amax'] == 100.0 or kwargs['amax'] == '100':
            if fullText[37][0] == '#':
                fullText[37] = fullText[37][1:]     # Remove the pound at 100.0
                fullText[30] = '#' + fullText[30]   # Add the pound at 0.25
            else:
                raise ValueError('JOB_OPTTHIN_CREATE: There is a comment problem at amax = 100!')
        elif kwargs['amax'] == 1000.0 or kwargs['amax'] == '1mm':
            if fullText[38][0] == '#':
                fullText[38] = fullText[38][1:]     # Remove the pound at 1mm
                fullText[30] = '#' + fullText[30]   # Add the pound at 0.25
            else:
                raise ValueError('JOB_OPTTHIN_CREATE: There is a comment problem at amax = 1mm!')
        else:
            raise ValueError('JOB_OPTTHIN_CREATE: Invalid input for AMAX!')
    
    # Now we can cycle through the easier changes desired:    
    if 'labelend' in kwargs:                        # Labelend for output files
        fullText[5] = (fullText[5][:14] + str(kwargs['labelend']) + 
                        fullText[5][-27:])
    if 'tstar' in kwargs:                           # Stellar effective temperature
        fullText[8] = (fullText[8][:10] + str(kwargs['tstar']) + 
                        fullText[8][-42:])
    if 'rstar' in kwargs:                           # Stellar radius (solar units)
        fullText[9] = (fullText[9][:10] + str(kwargs['rstar']) + 
                        fullText[9][-43:])
    if 'dist' in kwargs:                            # Distance (in pc)
        fullText[10] = (fullText[10][:15] + str(kwargs['dist']) + 
                        fullText[10][-25:])
    if 'mui' in kwargs:                             # Cosine of inclination angle
        fullText[13] = (fullText[13][:9] + str(kwargs['mui']) + 
                        fullText[13][-54:])
    if 'rout' in kwargs:                            # Outer radius
        fullText[14] = (fullText[14][:10] + str(kwargs['rout']) + 
                        fullText[14][-21:])
    if 'rin' in kwargs:                             # Inner radius
        fullText[15] = (fullText[15][:9] + str(kwargs['rin']) + 
                        fullText[15][-26:])
    if 'tau' in kwargs:                             # Optical depth (?)
        fullText[17] = (fullText[17][:12] + str(kwargs['tau']) + 
                        fullText[17][-4:])
    if 'power' in kwargs:                           # No idea what this one is, hah.
        fullText[18] = (fullText[18][:11] + str(kwargs['power']) + 
                        fullText[18][-2:])
    if 'fudgeorg' in kwargs:                        # No idea what this is either...
        fullText[19] = (fullText[19][:14] + str(kwargs['fudgeorg']) + 
                        fullText[19][-2:])
    if 'fudgetroi' in kwargs:                       # Or this...
        fullText[20] = (fullText[20][:15] + str(kwargs['fudgetroi']) + 
                        fullText[20][-2:])
    if 'fracsil' in kwargs:                         # Fraction of silicates by mass
        fullText[21] = (fullText[21][:13] + str(kwargs['fracsil']) + 
                        fullText[21][-4:])
    if 'fracent' in kwargs:                         # Fraction of enstatite by mass
        fullText[22] = (fullText[22][:13] + str(kwargs['fracent']) + 
                        fullText[22][-2:])
    if 'fracforst' in kwargs:                       # Fraction of forsterite by mass
        fullText[23] = (fullText[23][:15] + str(kwargs['fracforst']) + 
                        fullText[23][-2:])
    if 'fracamc' in kwargs:                         # Fraction of amorphous carbon by mass
        fullText[24] = (fullText[24][:13] + str(kwargs['fracamc']) + 
                        fullText[24][-2:])
    
    # Once all changes have been made, we just create a new optthin job file:
    if high:
        string_num = numCheck(jobn, high=True)
    else:
        string_num  = numCheck(jobn)
    newJob      = open(path+'job_optthin'+string_num, 'w')
    newJob.writelines(fullText)
    newJob.close()
    
    return

def model_rchi2(objname, model, path):
    """
    (By Dan)
    Calculates a reduced chi-squared goodness of fit.
    
    INPUTS
    objname: The name of the object to match for observational data.
    model: The model to test. Must be an instance of TTS_Model().
    path: The path containing the observations.
    
    OUTPUT
    rchi_sq: The value for the reduced chi-squared test on the model.
    
    """
    
    # Read in observations:
    objectObs   = loadPickle(objname, picklepath=path)
    
    # Get the model and observations onto the same wavelength vector:
    wavelength  = np.array([], dtype=float)
    flux        = np.array([], dtype=float)
    # Build the observations flux and wavelength vectors:
    for obsKey in objectObs.photometry.keys():
        if obsKey == 'DCT' or obsKey == 'DCT_Raw' or obsKey == 'DCT Raw':
            continue                            # Skip DCT Data
        if obsKey in objectObs.ulim:
            continue                            # Skip upper limits
        wavelength = np.append(wavelength, objectObs.photometry[obsKey]['wl'])
        flux    = np.append(flux, objectObs.photometry[obsKey]['lFl'])
    for specKey in objectObs.spectra.keys():
        wavelength = np.append(wavelength, objectObs.spectra[specKey]['wl'])
        flux    = np.append(flux, objectObs.spectra[specKey]['lFl'])
    waveindex   = np.argsort(wavelength)        # indices that sort the array
    wavelength  = wavelength[waveindex]
    flux        = flux[waveindex]
    modelFlux   = np.interp(wavelength, model.data['wl'], model.data['total'])
    
    # The tough part -- figuring out the proper weights. Let's take a stab:
    weights     = np.ones(len(wavelength))      # Start with all ones
    #weights[wavelength <= 8.5] = 25             # Some weight to early phot/spec data
    #weights[wavelength >= 22]  = 60             # More weight to outer piece of SED
    weights[wavelength <= 22]  = 75
    weights[wavelength <= 1] = 1
    
    # Calculate the reduced chi-squared value for the model:
    chi_arr     = (flux - modelFlux) * weights / flux
    rchi_sq     = np.sum(chi_arr*chi_arr) / (len(chi_arr) - 1.)
    
    return rchi_sq

#---------------------------------------------------CLASSES------------------------------------------------------
class TTS_Model(object):
    """
    (By Dan)
    Contains all the data and meta-data for a TTS Model from the D'Alessio et al. 2006 models. The input
    will come from fits files that are created via Connor's IDL collate procedure.
    
    ATTRIBUTES
    name: Name of the object (e.g., CVSO109, V410Xray-2, ZZ_Tau, etc.).
    jobn: The job number corresponding to this model.
    mstar: Star's mass.
    tstar: Star's effective temperature, based on Kenyon and Hartmann 1995.
    rstar: Star's radius.
    dist: Distance to the star.
    mdot: Mass accretion rate.
    alpha: Alpha parameter (from the viscous alpha disk model).
    mui: Inclination of the system.
    rdisk: The outer radius of the disk.
    amax: The "maximum" grain size in the disk. (or just suspended in the photosphere of the disk?)
    eps: The epsilon parameter, i.e., the amount of dust settling in the disk.
    tshock: The temperature of the shock at the stellar photosphere.
    temp: The temperature at the inner wall (1400 K maximum).
    altinh: Scale heights of extent of the inner wall.
    wlcut_an: 
    wlcut_sc: 
    nsilcomp: Number of silicate compounds.
    siltotab: Total silicate abundance.
    amorf_ol: 
    amorf_py: 
    forsteri: Forsterite Fractional abundance.
    enstatit: Enstatite Fractional abundance.
    rin: The inner radius in AU.
    data: The data for each component inside the model.
    
    METHODS
    __init__: initializes an instance of the class, and loads in the relevant data.
    calc_total: Calculates the "total" (combined) flux based on which components you want, then loads it into
                the data attribute under the key 'total'.
    
    """
    
    def __init__(self, name, jobn, dpath=datapath, full_trans=1, high=0, headonly=0):
        """
        (By Dan)
        Initializes instances of this class and loads the relevant data into attributes.
        
        INPUTS
        name: Name of the object being modeled. Must match naming convention used for models.
        jobn: Job number corresponding to the model being loaded into the object. Again, must match convention.
        full_trans: BOOLEAN -- if 1 (True) will load data as a full or transitional disk. If 0 (False), as a pre-trans. disk.
        high: BOOLEAN -- if 1 (True), the model file being read in has a 4-digit number string rather than 3-digit string.
        headonly: BOOLEAN -- if 1 (True) will only load the header metadata into the object, and will not load in the model data.
        
        """
        
        # First, sanity check:
        if full_trans != 0 and full_trans != 1:
            raise ValueError('__INIT__: full_trans is a boolean -- must be a 0 or 1!')
        # Read in the fits file:
        if high:
            stringnum   = numCheck(jobn, high=True)
        else:
            stringnum   = numCheck(jobn)                                # Convert jobn to the proper string format
        fitsname        = dpath + name + '_' + stringnum + '.fits'      # Fits filename, preceeded by the path from paths section
        HDUlist         = fits.open(fitsname)                           # Opens the fits file for use
        header          = HDUlist[0].header                             # Stores the header in this variable
        
        # The new Python version of collate flips array indices, so must identify which collate.py was used:
        if len(HDUlist[0].data['wl']) == 4:
            new         = 1
        else:
            new         = 0
        
        # Initialize meta-data attributes for this object:
        self.name       = name
        self.jobn       = jobn
        self.mstar      = header['MSTAR']
        self.tstar      = header['TSTAR']
        self.rstar      = header['RSTAR']
        self.dist       = header['DISTANCE']
        self.mdot       = header['MDOT']
        self.alpha      = header['ALPHA']
        self.mui        = header['MUI']
        self.rdisk      = header['RDISK']
        self.amax       = header['AMAXS']
        self.eps        = header['EPS']
        self.tshock     = header['TSHOCK']
        self.temp       = header['TEMP']
        self.altinh     = header['ALTINH']
        self.wlcut_an   = header['WLCUT_AN']
        self.wlcut_sc   = header['WLCUT_SC']
        self.nsilcomp   = header['NSILCOMP']
        self.siltotab   = header['SILTOTAB']
        self.amorf_ol   = header['AMORF_OL']
        self.amorf_py   = header['AMORF_PY']
        self.forsteri   = header['FORSTERI']
        self.enstatit   = header['ENSTATIT']
        self.rin        = header['RIN']
        self.dpath      = dpath
        
        # Initialize data attributes for this object using nested dictionaries:
        # wl is the wavelength (corresponding to all three flux arrays). Phot is the stellar photosphere emission.
        # iWall is the flux from the inner wall. Disk is the emission from the angle file.
        if headonly == 0:
            if full_trans:
                if new:
                    self.data   = {'wl': HDUlist[0].data[0,:], 'phot': HDUlist[0].data[1,:], 'iwall': HDUlist[0].data[2,:], \
                                   'disk': HDUlist[0].data[3,:]}
                else:
                    self.data   = {'wl': HDUlist[0].data[:,0], 'phot': HDUlist[0].data[:,1], 'iwall': HDUlist[0].data[:,2], \
                                   'disk': HDUlist[0].data[:,3]}
            else:
                # If a pre-transitional disk, have to match the job to the inner-wall job.
                z           = raw_input('What altinh value are you using for the inner wall? ')
                match       = searchJobs(name, dpath=dpath, amaxs=header['AMAXS'], eps=header['EPS'], alpha=header['ALPHA'], mdot=header['MDOT'], altinh=int(z),rdisk=1, temp=1400)
                if len(match) == 0:
                    raise IOError('__INIT__: No inner wall model matches these parameters.')
                elif len(match) >1:
                    raise IOError('__INIT__: Multiple inner wall models match. Do not know which one to pick.')
                else:
                    outfits = fits.open(dpath + name + '_' + match[0] + '.fits')
                    if new:
                        self.data   = {'wl': HDUlist[0].data[0,:], 'phot': HDUlist[0].data[1,:], 'owall': HDUlist[0].data[2,:], \
                                       'disk': HDUlist[0].data[3,:], 'iwall': outfits[0].data[2,:]}
                    else:
                        self.data   = {'wl': HDUlist[0].data[:,0], 'phot': HDUlist[0].data[:,1], 'owall': HDUlist[0].data[:,2], \
                                       'disk': HDUlist[0].data[:,3], 'iwall': outfits[0].data[:,2]}
            
        HDUlist.close()                                                 # Closes the fits file, since we no longer need it
        
    def calc_total(self, phot=1, wall=1, disk=1, owall=0, dust=0, verbose=1, dust_high=0):
        """
        (By Dan)
        Calculates the total flux for our object (likely to be used for plotting and/or analysis). Once calculated, it
        will be added to the data attribute for this object. If already calculated, will overwrite.
        
        INPUTS
        phot: BOOLEAN -- if 1 (True), will add photosphere component to the combined model.
        wall: BOOLEAN -- if 1 (True), will add inner wall component to the combined model.
        disk: BOOLEAN -- if 1 (True), will add disk component to the combined model.
        owall: BOOLEAN -- if 1 (True), will add outer wall component to the combined model (relevant for pre-trans only).
        dust: INTEGER -- Must correspond to an opt. thin dust model number linked to a fits file in datapath directory.
        
        """
        
        # Add the components to the total flux, checking each component along the way:
        totFlux         = np.zeros(len(self.data['wl']), dtype=float)
        if phot:
            if verbose:
                print 'CALC_TOTAL: Adding photosphere component to the total flux.'
            totFlux     = totFlux + self.data['phot']
        if wall:
            if verbose:
                print 'CALC_TOTAL: Adding inner wall component to the total flux.'
            totFlux     = totFlux + self.data['iwall']
        if disk:
            if verbose:
                print 'CALC_TOTAL: Adding disk component to the total flux.'
            totFlux     = totFlux + self.data['disk']
        if owall:
            if verbose:
                print 'CALC_TOTAL: Adding outer wall component to the total flux.'
            totFlux     = totFlux + self.data['owall']
        if dust != 0:
            try:
                dustNum = numCheck(dust, high=dust_high)
            except:
                raise ValueError('CALC_TOTAL: Error! Dust input not a valid integer')
                
            dustHDU     = fits.open(self.dpath+self.name+'_OTD_'+dustNum+'.fits')
            if verbose:
                print 'CALC_TOTAL: Adding optically thin dust component to total flux.'
            self.data['dust']   = dustHDU[0].data[1,:]
            totFlux     = totFlux + self.data['dust']
        
        # Add the total flux array to the data dictionary attribute:
        if verbose:
            print 'CALC_TOTAL: Total flux calculated. Adding to the data structure.'
        self.data['total'] = totFlux
        return
    
class TTS_Obs(object):
    """
    (By Dan)
    Contains all the observational data for a given target system. Allows you to create a pickle with the data, so it can
    be reloaded in at a future time without the need to re-initialize the object. However, to open up the pickle, you will
    need to have this source code where Python can access it.
    
    ATTRIBUTES
    name: The name of the target whose observations this represents.
    spectra: The spectra measurements for said target.
    photometry: The photometry measurements for said target.
    ulim: Which (if any) photometry points are upper limits.
    
    METHODS
    __init__: Initializes an instance of this class. Creates initial attributes (name and empty data dictionaries).
    add_spectra: Adds an entry (or replaces an entry) in the spectra attribute dictionary.
    add_photometry: Adds an entry (or replaces an entry) in the photometry attribute dictionary.
    SPPickle: Saves the object as a pickle to be reloaded later. This will not work if you've reloaded the module before saving.
    
    """
    
    def __init__(self, name):
        """
        (By Dan)
        Initializes instances of the class and loads in data to the proper attributes.
        
        INPUTS
        name: The name of the target for which the data represents.
        
        """
        # Initalize attributes as empty. Can add to the data later.
        self.name       = name
        self.spectra    = {}
        self.photometry = {}
        self.ulim       = []
        
    def add_spectra(self, scope, wlarr, fluxarr, errors=None):
        """
        (By Dan)
        Adds an entry to the spectra attribute.
        
        INPUTS
        scope: The telescope or instrument that the spectrum was taken with.
        wlarr: The wavelenth array of the data. Should be in microns. Note: this is not checked.
        fluxarr: The flux array of the data. Should be in erg s-1 cm-2. Note: this is not checked.
        errors: (optional) The array of flux errors. Should be in erg s-1 cm-2. If None (default), will not add.
        
        """
        
        # Check if the telescope data already exists in the data file:
        if scope in self.spectra.keys():
            print 'ADD_SPECTRA: Warning! This will overwrite current entry!'
            tries               = 1
            while tries <= 5:                                           # Give user 5 chances to choose if overwrite data or not
                proceed         = raw_input('Proceed? (Y/N): ')         # Prompt and collect manual answer - requires Y,N,Yes,No (not case sensitive)
                if proceed.upper() == 'Y' or proceed.upper() == 'YES':  # If Y or Yes, overwrite file, then break out of loop
                    print 'ADD_SPECTRA: Replacing entry.'
                    if errors == None:
                        self.spectra[scope] = {'wl': wlarr, 'lFl': fluxarr}
                    else:
                        self.spectra[scope] = {'wl': wlarr, 'lFl': fluxarr, 'err': errors}
                    break
                elif proceed.upper() == 'N' or proceed.upper() == 'NO': # If N or No, do not overwrite data and return
                    print 'ADD_SPECTRA: Will not replace entry. Returning now.'
                    return
                else:
                    tries       = tries + 1                             # If something else, lets you try again
            else:
                raise IOError('You did not enter the correct Y/N response. Returning without replacing.')   # If you enter bad response too many times, raise error.
        else:
            if errors == None:
                self.spectra[scope] = {'wl': wlarr, 'lFl': fluxarr}     # If not an overwrite, writes data to the object's spectra attribute dictionary.
            else:
                self.spectra[scope] = {'wl': wlarr, 'lFl': fluxarr, 'err': errors}
        return
    
    def add_photometry(self, scope, wlarr, fluxarr, errors=None, ulim=0):
        """
        (By Dan)
        Adds an entry to the photometry attribute.
        
        INPUTS
        scope: The telescope or instrument that the photometry was taken with.
        wlarr: The wavelength array of the data. Can also just be one value if an individual point. Should be in microns. Note: this is not checked.
        fluxarr: The flux array corresponding to the data. Should be in erg s-1 cm-2. Note: this is not checked.
        errors: (optional) The array of flux errors. Should be in erg s-1 cm-2. If None (default), will not add.
        ulim: BOOLEAN -- whether or not this photometric data is or is not an upper limit.
        
        """
        
        # Check if the telescope data already exists in the data file:
        if scope in self.photometry.keys():
            print 'ADD_PHOTOMETRY: Warning! This will overwrite current entry!'
            tries                   = 1
            while tries <= 5:                                               # Give user 5 chances to choose if overwrite data or not
                proceed             = raw_input('Proceed? (Y/N): ')         # Prompt and collect manual answer - requires Y,N,Yes,No (not case sensitive)
                if proceed.upper() == 'Y' or proceed.upper() == 'YES':      # If Y or Yes, overwrite file, then break out of loop
                    print 'ADD_PHOTOMETRY: Replacing entry.'
                    if errors == None:
                        self.photometry[scope]  = {'wl': wlarr, 'lFl': fluxarr}
                    else:
                        self.photometry[scope]  = {'wl': wlarr, 'lFl': fluxarr, 'err': errors}
                    if ulim == 1:
                        self.ulim.append(scope)                             # If upper limit, append metadata to ulim attribute list.
                    break
                elif proceed.upper() == 'N' or proceed.upper() == 'NO':     # If N or No, do not overwrite data and return
                    print 'ADD_PHOTOMETRY: Will not replace entry. Returning now.'
                    return
                else:
                    tries           = tries + 1                             # If something else, lets you try again
            else:
                raise IOError('You did not enter the correct Y/N response. Returning without replacing.')   # If you enter bad response too many times, raise error.
        else:
            if errors == None:
                self.photometry[scope]  = {'wl': wlarr, 'lFl': fluxarr}     # If not an overwrite, writes data to the object's photometry attribute dictionary.
            else:
                self.photometry[scope]  = {'wl': wlarr, 'lFl': fluxarr, 'err': errors}
            if ulim == 1:
                self.ulim.append(scope)                                     # If upper limit, append metadata to ulim attribute list.
        return
    
    def SPPickle(self, picklepath):
        """
        (By Dan)
        Saves the object as a pickle. Damn it Jim, I'm a doctor not a pickle farmer!
        
        WARNING: If you reload the module BEFORE you save the observations as a pickle, this will NOT work! I'm not
        sure how to go about fixing this issue, so just be aware of this.
        
        INPUTS
        picklepath: The path where you will save the pickle. I recommend datapath for simplicity.
        
        """
        # Check whether or not the pickle already exists:
        pathlist        = filelist(picklepath)
        outname         = self.name + '_obs.pkl'
        count           = 1
        while 1:
            if outname in pathlist:
                if count == 1:
                    print 'SPPICKLE: Pickle already exists in directory. For safety, will change name.'
                countstr= numCheck(count)
                count   = count + 1
                outname = self.name + '_obs_' + countstr + '.pkl'
            else:
                break
        # Now that that's settled, let's save the pickle.
        f               = open(picklepath + outname, 'wb')
        cPickle.dump(self, f)
        f.close()
        return



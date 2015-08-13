#!/usr/bin/env python
# Created by Dan Feldman and Connor Robinson for analyzing data from Espaillat Group research models.
# Last updated: 8/13/15 by Dan

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
edgepath        = '/Users/danfeldman/Python_Code/EDGE/'
datapath        = '/Users/danfeldman/Orion_Research/Orion_Research/CVSO_4Objs/Models/CVSO109PT2/'
figurepath      = '/Users/danfeldman/Orion_Research/Orion_Research/CVSO_4Objs/Look_SEDs/CVSO107/'
#figurepath      = '/Users/danfeldman/Orion_Research/Orion_Research/CVSO_4Objs/Models/Full_CVSO_Grid/CVSO58_sil/'

#---------------------------------------------INDEPENDENT FUNCTIONS----------------------------------------------
# A function is considered independent if it does not reference any other function or class in this module.

def keyErrHandle(func):
    """
    A decorator to allow methods and functions to have key errors, and to print the failed key.
    """
    
    def handler(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except KeyError as badKey:
            print('Key error was encountered. The missing key is: ' + str(badKey))
            return 0
        else:
            return 1
    return handler

def filelist(path):
    """
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

def deci_to_time(ra=None, dec=None):
    """    
    Converts decimal values of ra and dec into arc time coordinates.
    
    INPUTS
    ra: The float value of right ascension.
    dec: The float value of declination.
    
    OUTPUTS
    new_ra: The converted RA. If no ra supplied, returns -1
    new_dec: The converted dec. If no dec supplied, returns -1
    """
    
    new_ra  = -1
    new_dec = -1
    
    if ra is not None:
        if type(ra) != float:
            raise ValueError('DECI_TO_TIME: RA is not a float. Cannot convert.')  
          
        # First, we find the number of hours:
        hours    = ra / 15.0
        hoursInt = int(hours)
        hours    = hours - hoursInt
        
        # Next, we want minutes:
        minutes  = hours * 60.0
        minInt   = int(minutes)
        minutes  = minutes - minInt
        
        # Lastly, seconds:
        seconds  = minutes * 60.0
        new_ra   = '{0:02d} {1:02d} {2:.2f}'.format(hoursInt, minInt, seconds) 
    
    if dec is not None:
        if type(dec) != float:
            raise ValueError('DECI_TO_TIME: Dec is not a float. Cannot convert.')
        
        # For dec, have to check and store the sign:
        if dec < 0.0:
            sign = '-'
        else:
            sign = '+'
        dec      = abs(dec)
        
        # First, we find the number of degrees:
        degInt   = int(dec)
        deg      = dec - degInt
        
        # Next, we want minutes:
        minutes  = deg * 60.0
        minInt   = int(minutes)
        minutes  = minutes - minInt
        
        # Lastly, seconds:
        seconds  = minutes * 60.0
        new_dec  = '{0:s}{1:02d} {2:02d} {3:.2f}'.format(sign, degInt, minInt, seconds)
    
    return new_ra, new_dec

def convertFreq(value):
    """
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
        raise ValueError('CONVERTMAG: Unknown Band given. Cannot convert.')
    
    if jy == 'False':
        # Next, convert to flux from Janskys:
        flux        = convertJy(fluxJ, wavelength)              # Ok, so maybe this is a dependent function...
        return flux                                             # Shhhhhhh! :)
    return fluxJ

def numCheck(num, high=0):
    """
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
def look(obs, model=None, jobn=None, save=0, savepath=figurepath, colkeys=None, diskcomb=0, xlim=[2e-1, 2e3], ylim=[1e-15, 1e-9]):
    """
    Creates a plot of a model and the observations for a given target.
    
    INPUTS
    model: The object containing the target's model. Should be an instance of the TTS_Model class. This is an optional input.
    obs: The object containing the target's observations. Should be an instance of the TTS_Obs class.
    jobn: The "job number." This is meaningless for observation-only plots, but if you save the file, we require a number.
    save: BOOLEAN -- If 1 (True), will save the plot in a pdf file. If 0 (False), will output to screen.
    savepath: The path that a saved PDF file will be written to. This is defaulted to the hard-coded figurepath at top of this file.
    colkeys: An optional input array of color strings. This can be used to overwrite the normal color order convention. Options include:
             p == purple, r == red, m == magenta, b == blue, c == cyan, l == lime, t == teal, g == green, y == yellow, o == orange,
             k == black, w == brown, v == violet, d == gold, n == pumpkin, e == grape, j == jeans, s == salmon
             If not specified, the default order will be used, and once you run out, we'll have an error. So if you have more than 18
             data types, you'll need to supply the order you wish to use (and which to repeat). Or you can add new colors using html tags
             to the code, and then update this header.
    diskcomb: BOOLEAN -- If 1 (True), will combine outer wall and disk components into one for plotting. If 0 (False), will separate.
    xlim: A list containing the lower and upper x-axis limits, respectively. Has default values.
    ylim: A list containing the lower and upper y-axis limits, respectively. Has default values.
    
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
    plt.figure(1)
    
    # Plot the spectra first:
    for sind, skey in enumerate(speckeys):
        plt.plot(obs.spectra[skey]['wl'], obs.spectra[skey]['lFl'], color=colors[colkeys[sind]] , linewidth=2.0, label=skey)
    
    # Next is the photometry:
    for pind, pkey in enumerate(photkeys):
        # If an upper limit only:
        if pkey in obs.ulim:
            plt.plot(obs.photometry[pkey]['wl'], obs.photometry[pkey]['lFl'], 'v', \
                     color=colors[colkeys[pind+len(speckeys)]], markersize=7, label=pkey)
        # If not an upper limit, plot as normal:
        else:
            if 'err' not in obs.photometry[pkey].keys():
                plt.plot(obs.photometry[pkey]['wl'], obs.photometry[pkey]['lFl'], 'o', mfc='w', mec=colors[colkeys[pind+len(speckeys)]], mew=1.0,\
                         markersize=7, label=pkey)
            else:
                plt.errorbar(obs.photometry[pkey]['wl'], obs.photometry[pkey]['lFl'], yerr=obs.photometry[pkey]['err'], \
                             mec=colors[colkeys[pind+len(speckeys)]], fmt='o', mfc='w', mew=1.0, markersize=7, \
                             ecolor=colors[colkeys[pind+len(speckeys)]], elinewidth=2.0, capsize=2.0, label=pkey)
    
    # Now, the model (if a model supplied):
    if model != None:
        modkeys         = model.data.keys()
        if 'phot' in modkeys:
            plt.plot(model.data['wl'], model.data['phot'], ls='--', c='b', linewidth=2.0, label='Photosphere')
        try:
            plt.plot(model.data['wl'], model.newIWall, ls='--', c='#53EB3B', linewidth=2.0, label='Inner Wall')
        except AttributeError:
            if 'iwall' in modkeys:
                plt.plot(model.data['wl'], model.data['iwall'], ls='--', c='#53EB3B', linewidth=2.0, label='Inner Wall')
        if diskcomb:
            try:
                diskflux     = model.newOwall + model.data['disk']
            except AttributeError:
                try:
                    diskflux = model.data['owall'] + model.data['disk']
                except KeyError:
                    print 'LOOK: Error, tried to combine outer wall and disk components but one component is missing!'
                else:    
                    plt.plot(model.data['wl'], diskflux, ls='--', c='#8B0A1E', linewidth=2.0, label='Outer Disk')
        else:
            try:
                plt.plot(model.data['wl'], model.newOWall, ls='--', c='#E9B021', linewidth=2.0, label='Outer Wall')
            except AttributeError:
                if 'owall' in modkeys:
                    plt.plot(model.data['wl'], model.data['owall'], ls='--', c='#E9B021', linewidth=2.0, label='Outer Wall')
            if 'disk' in modkeys:
                plt.plot(model.data['wl'], model.data['disk'], ls='--', c='#8B0A1E', linewidth=2.0, label='Disk')
        if 'dust' in modkeys:
            plt.plot(model.data['wl'], model.data['dust'], ls='--', c='#F80303', linewidth=2.0, label='Opt. Thin Dust')
        if 'scatt' in modkeys:
            plt.plot(model.data['wl'], model.data['scatt'], ls='--', c='#7A6F6F', linewidth=2.0, label='Scattered Light')
        if 'total' in modkeys:
            plt.plot(model.data['wl'], model.data['total'], c='k', linewidth=2.0, label='Combined Model')
        # Now, the relevant meta-data:
        plt.figtext(0.60,0.88,'Eps = '+ str(model.eps), color='#010000', size='9')
        plt.figtext(0.80,0.88,'Alpha = '+ str(model.alpha), color='#010000', size='9')
        plt.figtext(0.60,0.82,'Amax = '+ str(model.amax), color='#010000', size='9')
        plt.figtext(0.60,0.85,'Rin = '+ str(model.rin), color='#010000', size='9')
        plt.figtext(0.80,0.85,'Rout = '+ str(model.rdisk), color='#010000', size='9')
        plt.figtext(0.60,0.79,'Altinh = '+ str(model.wallH), color='#010000', size='9')
        plt.figtext(0.80,0.82,'Mdot = '+ str(model.mdot), color='#010000', size='9')
        # If we have an outer wall height:
        try:
            plt.figtext(0.80,0.79,'AltinhOuter = '+ str(model.owallH), color='#010000', size='9')
        except AttributeError:
            plt.figtext(0.60,0.76,'IWall Temp = '+ str(model.temp), color='#010000', size='9')
        else:
            plt.figtext(0.60,0.76,'IWall Temp = '+ str(model.itemp), color='#010000', size='9')
            plt.figtext(0.80,0.76,'OWall Temp = '+ str(model.temp), color='#010000', size='9')
        
    # Lastly, the remaining parameters to plotting (mostly aesthetics):
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.ylabel(r'${\rm \lambda F_{\lambda}\; (erg\; s^{-1}\; cm^{-2})}$')
    plt.xlabel(r'${\rm {\bf \lambda}\; (\mu m)}$')
    plt.title(obs.name.upper())
    plt.legend(loc=3)
    
    # Should we save or should we plot?
    if save:
        if type(jobn) != int:
            raise ValueError('LOOK: Jobn must be an integer if you wish to save the plot.')
        jobstr          = numCheck(jobn)
        plt.savefig(savepath + obs.name.upper() + '_' + jobstr + '.pdf', dpi=350)
    else:
        plt.show()
    plt.clf()

    return

def searchJobs(target, dpath=datapath, **kwargs):
    """
    Searches through the job file outputs to determine which jobs (if any) matches the set of input parameters.
    
    INPUTS
    target: The name of the target we're checking against (e.g., cvso109, DMTau, etc.).
    **kwargs: Any keyword arguments (kwargs) supplied. These should correspond to the header filenames (not case sensitive). The code
              will loop through each of these kwargs and see if they all match.
    
    OUTPUTS
    job_matches: A numpy array containing all the jobs that matched the kwargs. Can be an empty array, single value array, or 
                 multivalued array. Will contain matches by their integer number.
    """
    
    job_matches         = np.array([], dtype='string')
    targList            = filelist(dpath)
    
    # Pop out all files that do not correspond to jobs:
    not_data            = []
    for f in targList:
        if f.startswith(target+'_') and f.endswith('.fits'):
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

def loadPickle(name, picklepath=datapath, num=None, red=0):
    """
    Loads in a pickle saved from the TTS_Obs class.
    
    INPUTS
    name: The name of the object whose observations are stored in the pickle.
    picklepath: The directory location of pickle. Default path is datapath, defined at top of this module.
    num: An optional number provided if there are multiple pickles for this object and you want to load a specific one.
    
    OUTPUT
    pickle: The object containing the data loaded in from the pickle.
    """
    
    if red:
        if num == None:
            # Check if there is more than one
            flist           = filelist(picklepath)
            if (name + '_red_1.pkl') in flist:
                print 'LOADPICKLE: Warning! There is more than one pickle file for this object! Make sure it is the right one!'
            f               = open(picklepath+name+'_red.pkl', 'rb')
            pickle          = cPickle.load(f)
            f.close()
        elif num != None:
            f               = open(picklepath+name+'_red_'+numCheck(num)+'.pkl', 'rb')
            pickle          = cPickle.load(f)
            f.close()
        return pickle
    else:
        if num == None:
            # Check if there is more than one
            flist           = filelist(picklepath)
            if (name + '_obs_1.pkl') in flist:
                print 'LOADPICKLE: Warning! There is more than one pickle file for this object! Make sure it is the right one!'
            f               = open(picklepath+name+'_obs.pkl', 'rb')
            pickle          = cPickle.load(f)
            f.close()
        elif num != None:
            f               = open(picklepath+name+'_obs_'+numCheck(num)+'.pkl', 'rb')
            pickle          = cPickle.load(f)
            f.close()
        return pickle

def job_file_create(jobnum, path, high=0, iwall=0, **kwargs):
    """
    Creates a new job file that is used by the D'Alessio Model.
    
    INPUTS
    jobnum: The job number used to name the output job file.
    path: The path containing the sample job file, and ultimately, the output.
    high: BOOLEAN -- if True (1), output will be jobXXXX instead of jobXXX.
    iwall: BOOLEAN -- if True (1), output will turn off switches so we just run as inner wall.
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
        currently supported. If any supplied kwargs are unused, it will print at the end.
    
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
    if fullText[42][0] == '#' or fullText[43][0] == '#':
        raise ValueError('JOB_FILE_CREATE: There is a comment problem at amax=0.25')
    if fullText[70][0] == '#' or fullText[71][0] == '#':
        raise ValueError('JOB_FILE_CREATE: There is a comment problem at eps=0.0001')
    
    # Now we run through the list of changes desired and change them:
    # If we want to change the maximum grain size (amaxs):
    if 'amaxs' in kwargs:
        amaxVal = kwargs['amaxs']
        del kwargs['amaxs']
        # amaxs is a commented out switch, so we need to know the desired size:
        if amaxVal == 0.25:
            pass
        elif amaxVal == 0.05:
            if fullText[36][0] == '#' and fullText[37][0] == '#':
                fullText[36] = fullText[36][1:]     # Remove the pound at 0.05
                fullText[37] = fullText[37][1:]
                fullText[42] = '#' + fullText[42]   # Add the pound at 0.25
                fullText[43] = '#' + fullText[43]
            else:
                raise ValueError('JOB_FILE_CREATE: There is a comment problem at amax=0.05')
        elif amaxVal == 0.1:
            if fullText[39][0] == '#' and fullText[40][0] == '#':
                fullText[39] = fullText[39][1:]     # Remove the pound at 0.1
                fullText[40] = fullText[40][1:]
                fullText[42] = '#' + fullText[42]   # Add the pound at 0.25
                fullText[43] = '#' + fullText[43]
            else:
                raise ValueError('JOB_FILE_CREATE: There is a comment problem at amax=0.1')
        elif amaxVal == 1.0:
            if fullText[45][0] == '#' and fullText[46][0] == '#':
                fullText[45] = fullText[45][1:]     # Remove the pound at 1.0
                fullText[46] = fullText[46][1:]
                fullText[42] = '#' + fullText[42]   # Add the pound at 0.25
                fullText[43] = '#' + fullText[43]
            else:
                raise ValueError('JOB_FILE_CREATE: There is a comment problem at amax=1.0')
        elif amaxVal == 2.0:
            if fullText[48][0] == '#' and fullText[49][0] == '#':
                fullText[48] = fullText[48][1:]     # Remove the pound at 2.0
                fullText[49] = fullText[49][1:]
                fullText[42] = '#' + fullText[42]   # Add the pound at 0.25
                fullText[43] = '#' + fullText[43]
            else:
                raise ValueError('JOB_FILE_CREATE: There is a comment problem at amax=2.0')
        elif amaxVal == 3.0:
            if fullText[51][0] == '#' and fullText[52][0] == '#':
                fullText[51] = fullText[51][1:]     # Remove the pound at 3.0
                fullText[52] = fullText[52][1:]
                fullText[42] = '#' + fullText[42]   # Add the pound at 0.25
                fullText[43] = '#' + fullText[43]
            else:
                raise ValueError('JOB_FILE_CREATE: There is a comment problem at amax=3.0')
        elif amaxVal == 4.0:
            if fullText[54][0] == '#' and fullText[55][0] == '#':
                fullText[54] = fullText[54][1:]     # Remove the pound at 4.0
                fullText[55] = fullText[55][1:]
                fullText[42] = '#' + fullText[42]   # Add the pound at 0.25
                fullText[43] = '#' + fullText[43]
            else:
                raise ValueError('JOB_FILE_CREATE: There is a comment problem at amax=4.0')
        elif amaxVal == 5.0:
            if fullText[57][0] == '#' and fullText[58][0] == '#':
                fullText[57] = fullText[57][1:]     # Remove the pound at 5.0
                fullText[58] = fullText[58][1:]
                fullText[42] = '#' + fullText[42]   # Add the pound at 0.25
                fullText[43] = '#' + fullText[43]
            else:
                raise ValueError('JOB_FILE_CREATE: There is a comment problem at amax=5.0')
        elif amaxVal == 10.0:
            if fullText[60][0] == '#' and fullText[61][0] == '#':
                fullText[60] = fullText[60][1:]     # Remove the pound at 10.0
                fullText[61] = fullText[61][1:]
                fullText[42] = '#' + fullText[42]   # Add the pound at 0.25
                fullText[43] = '#' + fullText[43]
            else:
                raise ValueError('JOB_FILE_CREATE: There is a comment problem at amax=10.0')
        elif amaxVal == 100.0:
            if fullText[63][0] == '#' and fullText[64][0] == '#':
                fullText[63] = fullText[63][1:]     # Remove the pound at 100.0
                fullText[64] = fullText[64][1:]
                fullText[42] = '#' + fullText[42]   # Add the pound at 0.25
                fullText[43] = '#' + fullText[43]
            else:
                raise ValueError('JOB_FILE_CREATE: There is a comment problem at amax=100.0')
        else:
            raise ValueError('JOB_FILE_CREATE: Invalid input for AMAXS!')
    
    # Now, we examine the epsilon parameter if a value provided:
    if 'epsilon' in kwargs:
        epsVal = kwargs['epsilon']
        del kwargs['epsilon']
        # Epsilon is a commented out switch, so we need the desired parameter:
        if epsVal == 0.0001:
            pass        # Default value is 0.0001
        elif epsVal == 0.001:
            if fullText[73][0] == '#' and fullText[74][0] == '#':
                fullText[73] = fullText[73][1:]     # Remove the pound at 0.001
                fullText[74] = fullText[74][1:]
                fullText[70] = '#' + fullText[70]   # Add the pound at 0.0001
                fullText[71] = '#' + fullText[71]
            else:
                raise ValueError('JOB_FILE_CREATE: There is a comment problem at eps=0.001')
        elif epsVal == 0.01:
            if fullText[76][0] == '#' and fullText[77][0] == '#':
                fullText[76] = fullText[76][1:]     # Remove the pound at 0.01
                fullText[77] = fullText[77][1:]
                fullText[70] = '#' + fullText[70]   # Add the pound at 0.0001
                fullText[71] = '#' + fullText[71]
            else:
                raise ValueError('JOB_FILE_CREATE: There is a comment problem at eps=0.01')
        elif epsVal == 0.1:
            if fullText[79][0] == '#' and fullText[80][0] == '#':
                fullText[79] = fullText[79][1:]     # Remove the pound at 0.1
                fullText[80] = fullText[80][1:]
                fullText[70] = '#' + fullText[70]   # Add the pound at 0.0001
                fullText[71] = '#' + fullText[71]
            else:
                raise ValueError('JOB_FILE_CREATE: There is a comment problem at eps=0.1')
        elif epsVal == 0.2:
            if fullText[82][0] == '#' and fullText[83][0] == '#':
                fullText[82] = fullText[82][1:]     # Remove the pound at 0.2
                fullText[83] = fullText[83][1:]
                fullText[70] = '#' + fullText[70]   # Add the pound at 0.0001
                fullText[71] = '#' + fullText[71]
            else:
                raise ValueError('JOB_FILE_CREATE: There is a comment problem at eps=0.2')
        elif epsVal == 0.5:
            if fullText[85][0] == '#' and fullText[86][0] == '#':
                fullText[85] = fullText[85][1:]     # Remove the pound at 0.5
                fullText[86] = fullText[86][1:]
                fullText[70] = '#' + fullText[70]   # Add the pound at 0.0001
                fullText[71] = '#' + fullText[71]
            else:
                raise ValueError('JOB_FILE_CREATE: There is a comment problem at eps=0.5')
        elif epsVal == 1.0:
            if fullText[88][0] == '#' and fullText[89][0] == '#':
                fullText[88] = fullText[88][1:]     # Remove the pound at 1.0
                fullText[89] = fullText[89][1:]
                fullText[70] = '#' + fullText[70]   # Add the pound at 0.0001
                fullText[71] = '#' + fullText[71]
            else:
                raise ValueError('JOB_FILE_CREATE: There is a comment problem at eps=1.0')
        else:
            raise ValueError('JOB_FILE_CREATE: Invalid input for epsilon!')
    
    # Now we can cycle through the easier changes desired:
    if 'mstar' in kwargs:                           # Stellar mass parameter
        mstarVal = kwargs['mstar']
        del kwargs['mstar']
        fullText[15] = fullText[15][:11] + str(mstarVal) + fullText[15][-11:]
    if 'tstar' in kwargs:                           # Photosphere temp parameter
        tstarVal = kwargs['tstar']
        del kwargs['tstar']
        fullText[16] = fullText[16][:11] + str(tstarVal) + fullText[16][-8:]
    if 'rstar' in kwargs:                           # Stellar radius parameter
        rstarVal = kwargs['rstar']
        del kwargs['rstar']
        fullText[17] = fullText[17][:11] + str(rstarVal) + fullText[17][-12:]
    if 'dist' in kwargs:                            # Stellar distance parameter
        distVal = kwargs['dist']
        del kwargs['dist']
        fullText[18] = fullText[18][:15] + str(distVal) + fullText[18][-14:]
    if 'mdot' in kwargs:                            # Accretion rate parameter
        mdotVal = kwargs['mdot']
        del kwargs['mdot']
        fullText[19] = fullText[19][:10] + str(mdotVal) + fullText[19][-15:]
    if 'tshock' in kwargs:                          # Shock temp parameter
        tshockVal = kwargs['tshock']
        del kwargs['tshock']
        fullText[22] = fullText[22][:11] + str(int(tshockVal)) + '.\n'
    if 'alpha' in kwargs:                           # Alpha viscosity parameter
        alphaVal = kwargs['alpha']
        del kwargs['alpha']
        fullText[25] = fullText[25][:11] + str(alphaVal) + fullText[25][-3:]
    if 'mui' in kwargs:                             # Cosine of inclination
        muiVal = kwargs['mui']
        del kwargs['mui']
        fullText[26] = fullText[26][:9] + str(muiVal) + fullText[26][-32:]
    if 'rdisk' in kwargs:                           # Outer disk radius parameter
        rdiskVal = kwargs['rdisk']
        del kwargs['rdisk']
        fullText[31] = fullText[31][:11] + str(rdiskVal) + fullText[31][-31:]
    if 'labelend' in kwargs:                        # Labelend on output files
        labelVal = kwargs['labelend']
        del kwargs['labelend']
        fullText[146] = fullText[146][:14] + str(labelVal) + fullText[146][-2:]
    if 'temp' in kwargs:                            # Inner wall temp parameter
        tempVal = kwargs['temp']
        del kwargs['temp']
        fullText[29] = fullText[29][:9] + str(int(tempVal)) + fullText[29][-50:]
    if 'altinh' in kwargs:                          # Inner wall height parameter
        altVal = kwargs['altinh']
        del kwargs['altinh']
        fullText[30] = fullText[30][:11] + str(altVal) + fullText[30][-24:]
    if iwall:
        # If an inner wall job is desired, turn off all but isilcom and iwalldust:
        fullText[153] = fullText[153][:-3] + '0' + fullText[153][-2:]   # IPHOT
        fullText[165] = fullText[165][:-3] + '0' + fullText[165][-2:]   # IOPA
        fullText[170] = fullText[170][:-3] + '0' + fullText[170][-2:]   # IVIS
        fullText[176] = fullText[176][:-3] + '0' + fullText[176][-2:]   # IIRR
        fullText[179] = fullText[179][:-5] + '0' + fullText[179][-4:]   # IPROP
        fullText[191] = fullText[191][:-3] + '0' + fullText[191][-2:]   # ISEDT
    
    # Once all changes have been made, we just create a new job file:
    if high:
        string_num = numCheck(jobnum, high=True)
    else:
        string_num  = numCheck(jobnum)
    newJob      = open(path+'job'+string_num, 'w')
    newJob.writelines(fullText)
    newJob.close()
    
    # Lastly, check for unused kwargs that may have been misspelled:
    if len(kwargs) != 0:
        print('JOB_FILE_CREATE: Unused kwargs, could be mistakes:')
        print kwargs.keys()
    
    return
    
def job_optthin_create(jobn, path, high=0, **kwargs):
    """
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
        currently supported. If any supplied kwargs are unused, it will print at the end.
    
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
        amaxVal = kwargs['amax']
        del kwargs['amax']
        # amax is a commented out switch, so we need to know the desired size:
        if amaxVal == 0.25:
            pass
        elif amaxVal == 0.05 or amaxVal == '0p05':
            if fullText[28][0] == '#':
                fullText[28] = fullText[28][1:]     # Remove the pound at 0.05
                fullText[30] = '#' + fullText[30]   # Add the pound at 0.25
            else:
                raise ValueError('JOB_OPTTHIN_CREATE: There is a comment problem at amax = 0.05!')
        elif amaxVal == 0.1 or amaxVal == '0p1':
            if fullText[29][0] == '#':
                fullText[29] = fullText[29][1:]     # Remove the pound at 0.1
                fullText[30] = '#' + fullText[30]   # Add the pound at 0.25
            else:
                raise ValueError('JOB_OPTTHIN_CREATE: There is a comment problem at amax = 0.1!')
        elif amaxVal == 1.0 or amaxVal == '1p0':
            if fullText[31][0] == '#':
                fullText[31] = fullText[31][1:]     # Remove the pound at 1.0
                fullText[30] = '#' + fullText[30]   # Add the pound at 0.25
            else:
                raise ValueError('JOB_OPTTHIN_CREATE: There is a comment problem at amax = 1.0!')
        elif amaxVal == 2.0 or amaxVal == '2p0':
            if fullText[32][0] == '#':
                fullText[32] = fullText[32][1:]     # Remove the pound at 2.0
                fullText[30] = '#' + fullText[30]   # Add the pound at 0.25
            else:
                raise ValueError('JOB_OPTTHIN_CREATE: There is a comment problem at amax = 2.0!')
        elif amaxVal == 3.0 or amaxVal == '3p0':
            if fullText[33][0] == '#':
                fullText[33] = fullText[33][1:]     # Remove the pound at 3.0
                fullText[30] = '#' + fullText[30]   # Add the pound at 0.25
            else:
                raise ValueError('JOB_OPTTHIN_CREATE: There is a comment problem at amax = 3.0!')
        elif amaxVal == 4.0 or amaxVal == '4p0':
            if fullText[34][0] == '#':
                fullText[34] = fullText[34][1:]     # Remove the pound at 4.0
                fullText[30] = '#' + fullText[30]   # Add the pound at 0.25
            else:
                raise ValueError('JOB_OPTTHIN_CREATE: There is a comment problem at amax = 4.0!')
        elif amaxVal == 5.0 or amaxVal == '5p0':
            if fullText[35][0] == '#':
                fullText[35] = fullText[35][1:]     # Remove the pound at 5.0
                fullText[30] = '#' + fullText[30]   # Add the pound at 0.25
            else:
                raise ValueError('JOB_OPTTHIN_CREATE: There is a comment problem at amax = 5.0!')
        elif amaxVal == 10.0 or amaxVal == '10':
            if fullText[36][0] == '#':
                fullText[36] = fullText[36][1:]     # Remove the pound at 10.0
                fullText[30] = '#' + fullText[30]   # Add the pound at 0.25
            else:
                raise ValueError('JOB_OPTTHIN_CREATE: There is a comment problem at amax = 10!')
        elif amaxVal == 100.0 or amaxVal == '100':
            if fullText[37][0] == '#':
                fullText[37] = fullText[37][1:]     # Remove the pound at 100.0
                fullText[30] = '#' + fullText[30]   # Add the pound at 0.25
            else:
                raise ValueError('JOB_OPTTHIN_CREATE: There is a comment problem at amax = 100!')
        elif amaxVal == 1000.0 or amaxVal == '1mm':
            if fullText[38][0] == '#':
                fullText[38] = fullText[38][1:]     # Remove the pound at 1mm
                fullText[30] = '#' + fullText[30]   # Add the pound at 0.25
            else:
                raise ValueError('JOB_OPTTHIN_CREATE: There is a comment problem at amax = 1mm!')
        else:
            raise ValueError('JOB_OPTTHIN_CREATE: Invalid input for AMAX!')
    
    # Now we can cycle through the easier changes desired:    
    if 'labelend' in kwargs:                        # Labelend for output files
        labelVal = kwargs['labelend']
        del kwargs['labelend']
        fullText[5] = fullText[5][:14] + str(labelVal) + fullText[5][-27:]
    if 'tstar' in kwargs:                           # Stellar effective temperature
        tstarVal = kwargs['tstar']
        del kwargs['tstar']
        fullText[8] = fullText[8][:10] + str(tstarVal) + fullText[8][-42:]
    if 'rstar' in kwargs:                           # Stellar radius (solar units)
        rstarVal = kwargs['rstar']
        del kwargs['rstar']
        fullText[9] = fullText[9][:10] + str(rstarVal) + fullText[9][-43:]
    if 'dist' in kwargs:                            # Distance (in pc)
        distVal = kwargs['dist']
        del kwargs['dist']
        fullText[10] = fullText[10][:15] + str(distVal) + fullText[10][-25:]
    if 'mui' in kwargs:                             # Cosine of inclination angle
        muiVal = kwargs['mui']
        del kwargs['mui']
        fullText[13] = fullText[13][:9] + str(muiVal) + fullText[13][-54:]
    if 'rout' in kwargs:                            # Outer radius
        routVal = kwargs['rout']
        del kwargs['rout']
        fullText[14] = fullText[14][:10] + str(routVal) + fullText[14][-21:]
    if 'rin' in kwargs:                             # Inner radius
        rinVal = kwargs['rin']
        del kwargs['rin']
        fullText[15] = fullText[15][:9] + str(rinVal) + fullText[15][-26:]
    if 'tau' in kwargs:                             # Optical depth
        tauVal = kwargs['tau']
        del kwargs['tau']
        fullText[17] = fullText[17][:12] + str(tauVal) + fullText[17][-4:]
    if 'power' in kwargs:                           # No idea what this one is, hah.
        powVal = kwargs['power']
        del kwargs['power']
        fullText[18] = fullText[18][:11] + str(powVal) + fullText[18][-2:]
    if 'fudgeorg' in kwargs:                        # No idea what this is either...
        orgVal = kwargs['fudgeorg']
        del kwargs['fudgeorg']
        fullText[19] = fullText[19][:14] + str(orgVal) + fullText[19][-2:]
    if 'fudgetroi' in kwargs:                       # Or this...
        troiVal = kwargs['fudgetroi']
        del kwargs['fudgetroi']
        fullText[20] = fullText[20][:15] + str(troiVal) + fullText[20][-2:]
    if 'fracsil' in kwargs:                         # Fraction of silicates by mass
        fsilVal = kwargs['fracsil']
        del kwargs['fracsil']
        fullText[21] = fullText[21][:13] + str(fsilVal) + fullText[21][-4:]
    if 'fracent' in kwargs:                         # Fraction of enstatite by mass
        fentVal = kwargs['fracent']
        del kwargs['fracent']
        fullText[22] = fullText[22][:13] + str(fentVal) + fullText[22][-2:]
    if 'fracforst' in kwargs:                       # Fraction of forsterite by mass
        forstVal = kwargs['fracforst']
        del kwargs['fracforst']
        fullText[23] = fullText[23][:15] + str(forstVal) + fullText[23][-2:]
    if 'fracamc' in kwargs:                         # Fraction of amorphous carbon by mass
        famcVal = kwargs['fracamc']
        del kwargs['fracamc']
        fullText[24] = fullText[24][:13] + str(famcVal) + fullText[24][-2:]
    
    # Once all changes have been made, we just create a new optthin job file:
    if high:
        string_num = numCheck(jobn, high=True)
    else:
        string_num  = numCheck(jobn)
    newJob      = open(path+'job_optthin'+string_num, 'w')
    newJob.writelines(fullText)
    newJob.close()
    
    # Lastly, check for unused kwargs that may have been misspelled:
    if len(kwargs) != 0:
        print('JOB_OPTTHIN_CREATE: Unused kwargs, could be mistakes:')
        print kwargs.keys()
    
    return

def model_rchi2(objname, model, path):
    """
    Calculates a reduced chi-squared goodness of fit.
    
    INPUTS
    objname: The name of the object to match for observational data.
    model: The model to test. Must be an instance of TTS_Model(), with a calculated total.
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
    # If there are any NaNs in the array, we need to chop them out, stat!
    if np.isnan(np.sum(flux)):
        badVals = np.where(np.isnan(flux))      # Where the NaNs are located
        flux    = np.delete(flux, badVals)
        wavelength = np.delete(wavelength, badVals)
    # Interpolate so the observations and model are on the same grid:
    modelFlux   = np.interp(wavelength, model.data['wl'], model.data['total'])
    
    # The tough part -- figuring out the proper weights. Let's take a stab:
    weights     = np.ones(len(wavelength))      # Start with all ones
    weights[wavelength <= 8.5] = 25             # Some weight to early phot/spec data
    weights[wavelength >= 22]  = 60             # More weight to outer piece of SED
    
    #weights[wavelength <= 22]  = 75            # These weights good for opt. thin dust comparisons
    #weights[wavelength <= 1] = 1
    
    # Calculate the reduced chi-squared value for the model:
    chi_arr     = (flux - modelFlux) * weights / flux
    rchi_sq     = np.sum(chi_arr*chi_arr) / (len(chi_arr) - 1.)
    
    return rchi_sq

#---------------------------------------------------CLASSES------------------------------------------------------
class TTS_Model(object):
    """
    Contains all the data and meta-data for a TTS Model from the D'Alessio et al. 2006 models. The input
    will come from fits files that are created via Connor's collate.py.
    
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
    dpath: Path where the data files are located.
    high: Whether or not the data was part of a 1000+ grid.
    data: The data for each component inside the model.
    extcorr: The self-extinction correction. If not carried out, saved as None.
    new: Whether or not the model was made with the newer version of collate.py.
    newIWall: The flux of an inner wall with a higher/lower altinh value.
    wallH: The inner wall height used by the look() function in plotting.
    
    METHODS
    __init__: Initializes an instance of the class, and loads in the relevant metadata.
    dataInit: Loads in the data to the object.
    calc_total: Calculates the "total" (combined) flux based on which components you want, then loads it into
                the data attribute under the key 'total'.
    """
    
    def __init__(self, name, jobn, dpath=datapath, high=0):
        """
        Initializes instances of this class and loads the relevant data into attributes.
        
        INPUTS
        name: Name of the object being modeled. Must match naming convention used for models.
        jobn: Job number corresponding to the model being loaded into the object. Again, must match convention.
        full_trans: BOOLEAN -- if 1 (True) will load data as a full or transitional disk. If 0 (False), as a pre-trans. disk.
        high: BOOLEAN -- if 1 (True), the model file being read in has a 4-digit number string rather than 3-digit string.
        """
        
        # Read in the fits file:
        if high:
            stringnum   = numCheck(jobn, high=1)
        else:
            stringnum   = numCheck(jobn)                                # Convert jobn to the proper string format
        fitsname        = dpath + name + '_' + stringnum + '.fits'      # Fits filename, preceeded by the path from paths section
        HDUlist         = fits.open(fitsname)                           # Opens the fits file for use
        header          = HDUlist[0].header                             # Stores the header in this variable
        
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
        self.high       = high
        self.extcorr    = None
        
        HDUlist.close()
        return
    
    def dataInit(self):
        """
        Initialize data attributes for this object using nested dictionaries:
        wl is the wavelength (corresponding to all three flux arrays). Phot is the stellar photosphere emission.
        iWall is the flux from the inner wall. Disk is the emission from the angle file. Scatt is the scattered
        light emission. Loads in self-extinction array if available.
        """
        
        stringnum    = numCheck(self.jobn, high=self.high)
        fitsname     = self.dpath + self.name + '_' + stringnum + '.fits'
        HDUdata      = fits.open(fitsname)
        header       = HDUdata[0].header
        
        # The new Python version of collate flips array indices, so must identify which collate.py was used:
        if 'EXTAXIS' in header.keys() or 'NOEXT' in header.keys():
            self.new = 1
        else:
            self.new = 0

        if self.new:
            # We will load in the components piecemeal based on the axes present in the header.
            # First though, we initialize with the wavelength array, since it's always present:
            self.data = {'wl': HDUdata[0].data[header['WLAXIS'],:]}
            
            # Now we can loop through the remaining possibilities:
            if 'PHOTAXIS' in header.keys():
                self.data['phot'] = HDUdata[0].data[header['PHOTAXIS'],:]
            else:
                print('DATAINIT: Warning: No photosphere data found for ' + self.name)
            if 'WALLAXIS' in header.keys():
                self.data['iwall'] = HDUdata[0].data[header['WALLAXIS'],:]
            else:
                print('DATAINIT: Warning: No outer wall data found for ' + self.name)
            if 'ANGAXIS' in header.keys():
                self.data['disk'] = HDUdata[0].data[header['ANGAXIS'],:]
            else:
                print('DATAINIT: Warning: No outer disk data found for ' + self.name)
            # Remaining components are not always (or almost always) present, so no warning given if missing!
            if 'SCATAXIS' in header.keys():
                self.data['scatt'] = HDUdata[0].data[header['SCATAXIS'],:]
                negScatt = np.where(self.data['scatt'] < 0.0)[0]
                if len(negScatt) > 0:
                    print('DATAINIT: WARNING: Some of your scattered light values are negative!')
            if 'EXTAXIS' in header.keys():
                self.extcorr       = HDUdata[0].data[header['EXTAXIS'],:]
        else:
            self.data = {'wl': HDUdata[0].data[:,0], 'phot': HDUdata[0].data[:,1], 'iwall': HDUdata[0].data[:,2], \
                         'disk': HDUdata[0].data[:,3]}
        
        HDUdata.close()
        return
    
    @keyErrHandle
    def calc_total(self, phot=1, wall=1, disk=1, dust=0, verbose=1, dust_high=0, altinh=None, save=0):
        """
        Calculates the total flux for our object (likely to be used for plotting and/or analysis). Once calculated, it
        will be added to the data attribute for this object. If already calculated, will overwrite.
        
        INPUTS
        phot: BOOLEAN -- if 1 (True), will add photosphere component to the combined model.
        wall: BOOLEAN -- if 1 (True), will add inner wall component to the combined model.
        disk: BOOLEAN -- if 1 (True), will add disk component to the combined model.
        dust: INTEGER -- Must correspond to an opt. thin dust model number linked to a fits file in datapath directory.
        verbose: BOOLEAN -- if 1 (True), will print messages of what it's doing.
        dust_high: BOOLEAN -- if 1 (True), will look for a 4 digit valued dust file.
        altinh: FLOAT/INT -- if not None, will multiply inner wall flux by that amount.
        save: BOOLEAN -- if 1 (True), will print out the components to a .dat file.
        """
        
        # Add the components to the total flux, checking each component along the way:
        totFlux         = np.zeros(len(self.data['wl']), dtype=float)
        componentNumber = 1
        if self.extcorr != None:
            componentNumber += 1
        if phot:
            if verbose:
                print 'CALC_TOTAL: Adding photosphere component to the total flux.'
            totFlux     = totFlux + self.data['phot']
            componentNumber += 1
        if wall:
            if verbose:
                print 'CALC_TOTAL: Adding inner wall component to the total flux.'
            if altinh != None:
                self.newIWall = self.data['iwall'] * altinh
                totFlux       = totFlux + self.newIWall     # Note: if save=1, will save iwall w/ the original altinh.
                self.wallH    = self.altinh * altinh
            else:
                totFlux       = totFlux + self.data['iwall']
                self.wallH    = self.altinh                 # Redundancy for plotting purposes.
                # If we tried changing altinh but want to now plot original, deleting the "newIWall" attribute from before.
                try:
                    del self.newIWall
                except AttributeError:
                    pass
            componentNumber += 1
        if disk:
            if verbose:
                print 'CALC_TOTAL: Adding disk component to the total flux.'
            totFlux     = totFlux + self.data['disk']
            componentNumber += 1
        if dust != 0:
            try:
                dustNum = numCheck(dust, high=dust_high)
            except:
                raise ValueError('CALC_TOTAL: Error! Dust input not a valid integer')
            dustHDU     = fits.open(self.dpath+self.name+'_OTD_'+dustNum+'.fits')
            if verbose:
                print 'CALC_TOTAL: Adding optically thin dust component to total flux.'
            if self.new:
                self.data['dust']   = dustHDU[0].data[1,:]
            else:
                self.data['dust']   = dustHDU[0].data[:,1]
            totFlux     = totFlux + self.data['dust']
            componentNumber += 1
        
        # If scattered emission is in the dictionary, add it:
        if 'scatt' in self.data.keys():
            scatt       = 1
            if verbose:
                print('CALC_TOTAL: Adding scattered light component to the total flux.')
            totFlux     = totFlux + self.data['scatt']
            componentNumber += 1
        
        # Add the total flux array to the data dictionary attribute:
        if verbose:
            print 'CALC_TOTAL: Total flux calculated. Adding to the data structure.'
        self.data['total'] = totFlux
        componentNumber += 1
        
        # If save, create an output file with these components printed out:
        if save:
            outputTable = np.zeros([len(totFlux), componentNumber])

            # Populate the header and data table with the components and names:
            headerStr   = 'Wavelength, Total Flux, '
            outputTable[:, 0] = self.data['wl']
            outputTable[:, 1] = self.data['total']
            colNum      = 2
            if phot:
                headerStr += 'Photosphere, '
                outputTable[:, colNum] = self.data['phot']
                colNum += 1
            if wall:
                headerStr += 'Inner Wall, '
                outputTable[:, colNum] = self.data['iwall']
                colNum += 1
            if disk:
                headerStr += 'Outer Disk, '
                outputTable[:, colNum] = self.data['disk']
                colNum += 1
            if dust != 0:
                headerStr += 'Opt. Thin Dust, '
                outputTable[:, colNum] = self.data['dust']
                colNum += 1
            if scatt:
                headerStr += 'Scattered Light, '
                outputTable[:, colNum] = self.data['scatt']
                colNum += 1
            if self.extcorr != None:
                headerStr += 'Tau, '
                outputTable[:, colNum] = self.extcorr
            
            # Trim the header and save:
            headerStr  = headerStr[0:-2]
            filestring = '%s%s_%s.dat' % (self.dpath, self.name, numCheck(self.jobn, high=self.high))
            np.savetxt(filestring, outputTable, fmt='%.3e', delimiter=', ', header=headerStr, comments='#')
        
        return

class PTD_Model(TTS_Model):
    """
    Contains all the data and meta-data for a PTD Model from the D'Alessio et al. 2006 models. The input
    will come from fits files that are created via Connor's collate.py.
    
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
    temp: The temperature at the outer wall component of the model.
    itemp: The temperature of the inner wall component of the model.
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
    dpath: Path where the data files are located.
    high: Whether or not the data was part of a 1000+ grid.
    data: The data for each component inside the model.
    extcorr: The self-extinction correction. If not carried out, saved as None.
    new: Whether or not the model was made with the newer version of collate.py.
    newIWall: The flux of an inner wall with a higher/lower altinh value.
    newOWall: The flux of an outer wall with a higher/lower altinh value.
    iwallH: The inner wall height used by the look() function in plotting.
    wallH: The outer wall height used by the look() function in plotting.
    
    METHODS
    __init__: initializes an instance of the class, and loads in the relevant metadata. No change.
    dataInit: Loads in the relevant data to the object. This differs from that of TTS_Model.
    calc_total: Calculates the "total" (combined) flux based on which components you want, then loads it into
                the data attribute under the key 'total'. This also differs from TTS_Model.
    """
    
    def dataInit(self, altname=None, jobw=None, highWall=0, **searchKwargs):
        """
        Initialize data attributes for this object using nested dictionaries:
        wl is the wavelength (corresponding to all three flux arrays). Phot is the stellar photosphere emission.
        iwall is the flux from the inner wall. Disk is the emission from the angle file. owall is the flux from
        the outer wall. Scatt is the scattered light emission. Also adds self-extinction array if available.
        
        You should either supply the job number of the inner wall file, or the kwargs used to find it via a
        search. Jobw 
        
        INPUTS
        altname: An alternate name for the inner wall file if necessary.
        jobw: The job number of the wall. Can be a string of 'XXX' or 'XXXX' based on the filename, or just the integer.
        highWall: BOOLEAN -- if 1, it means it's a 4 digit string rather than 3. Needed if you provide an int for jobw.
        **searchkwargs: Kwargs corresponding to parameters in the header that can be used to find the jobw value if 
                        you don't already know it. Otherwise, not necessary for the function call.
        """
        
        if jobw == None and len(searchKwargs) == 0:
            raise IOError('DATAINIT: You must enter either a job number or kwargs to match or search for an inner wall.')
        
        if jobw != None:
            # If jobw is an integer, make into a string:
            if type(jobw) == int:
                jobw = numCheck(jobw, high=highWall)
            
            # The case in which you supplied the job number of the inner wall:
            if altname == None:
                fitsname  = self.dpath + self.name + '_' + jobw + '.fits'
                HDUwall   = fits.open(fitsname)
            else:
                fitsname  = self.dpath + altname + '_' + jobw + '.fits'
                HDUwall   = fits.open(fitsname)
            
            # Make sure the inner wall job you supplied is, in fact, an inner wall.
            if 'NOEXT' not in HDUwall[0].header.keys():
                raise IOError('DATAINIT: Job you supplied is not an inner wall or needs to be collated again!')
            
            # Now, load in the disk data:
            stringNum     = numCheck(self.jobn, high=self.high)
            HDUdata       = fits.open(self.dpath + self.name + '_' + stringNum + '.fits')
            header        = HDUdata[0].header
            
            # Check if it's an old version or a new version:
            if 'EXTAXIS' in header.keys() or 'NOEXT' in header.keys():
                self.new  = 1
            else:
                self.new  = 0
            
            # Define the inner wall height.
            self.iwallH   = HDUwall[0].header['ALTINH']
            self.itemp   = HDUwall[0].header['TEMP']
            
            # Depending on old or new version is how we will load in the data. We require the wall be "new":
            if self.new:
                # Correct for self extinction:
                iwallFcorr= HDUwall[0].data[HDUwall[0].header['WALLAXIS'],:]*np.exp(-1*HDUdata[0].data[header['EXTAXIS'],:])
                
                # We will load in the components piecemeal based on the axes present in the header.
                # First though, we initialize with the wavelength and wall, since they're always present.
                self.data = {'wl': HDUdata[0].data[header['WLAXIS'],:], 'iwall': iwallFcorr}
                
                # Now we can loop through the remaining possibilities:
                if 'PHOTAXIS' in header.keys():
                    self.data['phot'] = HDUdata[0].data[header['PHOTAXIS'],:]
                else:
                    print('DATAINIT: Warning: No photosphere data found for ' + self.name)
                if 'WALLAXIS' in header.keys():
                    self.data['owall']= HDUdata[0].data[header['WALLAXIS'],:]
                else:
                    print('DATAINIT: Warning: No outer wall data found for ' + self.name)
                if 'ANGAXIS' in header.keys():
                    self.data['disk'] = HDUdata[0].data[header['ANGAXIS'],:]
                else:
                    print('DATAINIT: Warning: No outer disk data found for ' + self.name)
                # Remaining components are not always (or almost always) present, so no warning given if missing!
                if 'SCATAXIS' in header.keys():
                    self.data['scatt']= HDUdata[0].data[header['SCATAXIS'],:]
                    negScatt = np.where(self.data['scatt'] < 0.0)[0]
                    if len(negScatt) > 0:
                        print('DATAINIT: WARNING: Some of your scattered light values are negative!')
                if 'EXTAXIS' in header.keys():
                    self.extcorr      = HDUdata[0].data[header['EXTAXIS'],:]
            else:
                self.data = ({'wl': HDUdata[0].data[:,0], 'phot': HDUdata[0].data[:,1], 'owall': HDUdata[0].data[:,2],
                              'disk': HDUdata[0].data[:,3], 'iwall': HDUwall[0].data[HDUwall[0].header['WALLAXIS'],:]})
        
        else:
            # When doing the searchJobs() call, use **searchKwargs to pass that as the keyword arguments to searchJobs!
            if altname == None:
                match = searchJobs(self.name, dpath=self.dpath, **searchKwargs)
            else:
                match = searchJobs(altname, dpath=self.dpath, **searchKwargs)
            if len(match) == 0:
                raise IOError('DATAINIT: No inner wall model matches these parameters!')
            elif len(match) > 1:
                raise IOError('DATAINIT: Multiple inner wall models match. Do not know which one to pick.')
            else:
                if altname == None:
                    fitsname = self.dpath + self.name + '_' + match[0] + '.fits'
                else:
                    fitsname = self.dpath + altname + '_' + match[0] + '.fits'
                HDUwall  = fits.open(fitsname)
                
                # Make sure the inner wall job you supplied is, in fact, an inner wall.
                if 'NOEXT' not in HDUwall[0].header.keys():
                    raise IOError('DATAINIT: Job found is not an inner wall or needs to be collated again!')
            
                # Now, load in the disk data:
                stringNum    = numCheck(self.jobn, high=self.high)
                HDUdata      = fits.open(self.dpath + self.name + '_' + stringNum + '.fits')
                header       = HDUdata[0].header
            
                # Check if it's an old version or a new version:
                if 'EXTAXIS' in header.keys() or 'NOEXT' in header.keys():
                    self.new = 1
                else:
                    self.new = 0
                
                # Define the inner wall height:
                self.iwallH  = HDUwall[0].header['ALTINH']
                self.itemp   = HDUwall[0].header['TEMP']
                
                # Depending on old or new version is how we will load in the data. We require the wall be "new":
                if self.new:
                    # Correct for self extinction:
                    iwallFcorr = HDUwall[0].data[HDUwall[0].header['WALLAXIS'],:]*np.exp(-1*HDUdata[0].data[header['EXTAXIS'],:])
                    
                    # We will load in the components piecemeal based on the axes present in the header.
                    # First though, we initialize with the wavelength and wall, since they're always present:
                    self.data  = {'wl': HDUdata[0].data[header['WLAXIS'],:], 'iwall': iwallFcorr}
                
                    # Now we can loop through the remaining possibilities:
                    if 'PHOTAXIS' in header.keys():
                        self.data['phot'] = HDUdata[0].data[header['PHOTAXIS'],:]
                    else:
                        print('DATAINIT: Warning: No photosphere data found for ' + self.name)
                    if 'WALLAXIS' in header.keys():
                        self.data['owall']= HDUdata[0].data[header['WALLAXIS'],:]
                    else:
                        print('DATAINIT: Warning: No outer wall data found for ' + self.name)
                    if 'ANGAXIS' in header.keys():
                        self.data['disk'] = HDUdata[0].data[header['ANGAXIS'],:]
                    else:
                        print('DATAINIT: Warning: No outer disk data found for ' + self.name)
                    # Remaining components are not always (or almost always) present, so no warning given if missing!
                    if 'SCATAXIS' in header.keys():
                        self.data['scatt']= HDUdata[0].data[header['SCATAXIS'],:]
                        negScatt = np.where(self.data['scatt'] < 0.0)[0]
                        if len(negScatt) > 0:
                            print('DATAINIT: WARNING: Some of your scattered light values are negative!')
                    if 'EXTAXIS' in header.keys():
                        self.extcorr      = HDUdata[0].data[header['EXTAXIS'],:]
                else:
                    self.data = ({'wl': HDUdata[0].data[:,0], 'phot': HDUdata[0].data[:,1], 'owall': HDUdata[0].data[:,2],
                                  'disk': HDUdata[0].data[:,3], 'iwall': HDUwall[0].data[HDUwall[0].header['WALLAXIS'],:]})
        HDUdata.close()
        return
    
    def calc_total(self, phot=1, wall=1, disk=1, owall=1, dust=0, verbose=1, dust_high=0, altInner=None, altOuter=None, save=0):
        """
        Calculates the total flux for our object (likely to be used for plotting and/or analysis). Once calculated, it
        will be added to the data attribute for this object. If already calculated, will overwrite.
        
        INPUTS
        phot: BOOLEAN -- if 1 (True), will add photosphere component to the combined model.
        wall: BOOLEAN -- if 1 (True), will add inner wall component to the combined model.
        disk: BOOLEAN -- if 1 (True), will add disk component to the combined model.
        owall: BOOLEAN -- if 1 (True), will add outer wall component to the combined model.
        dust: INTEGER -- Must correspond to an opt. thin dust model number linked to a fits file in datapath directory.
        verbose: BOOLEAN -- if 1 (True), will print messages of what it's doing.
        dust_high: BOOLEAN -- if 1 (True), will look for a 4 digit valued dust file.
        altInner: FLOAT/INT -- if not None, will multiply inner wall flux by that amount.
        altOuter: FLOAT/INT -- if not None, will multiply outer wall flux by that amount.
        save: BOOLEAN -- if 1 (True), will print out the components to a .dat file.
        """
        
        # Add the components to the total flux, checking each component along the way:
        totFlux         = np.zeros(len(self.data['wl']), dtype=float)
        componentNumber = 1
        if phot:
            if verbose:
                print 'CALC_TOTAL: Adding photosphere component to the total flux.'
            totFlux     = totFlux + self.data['phot']
            componentNumber += 1
        if wall:
            if verbose:
                print 'CALC_TOTAL: Adding inner wall component to the total flux.'
            if altInner != None:
                self.newIWall = self.data['iwall'] * altInner
                totFlux       = totFlux + self.newIWall     # Note: if save=1, will save iwall w/ the original altinh.
                self.wallH    = self.iwallH * altInner
            else:
                totFlux = totFlux + self.data['iwall']
                self.wallH    = self.iwallH                 # Redundancy for plotting purposes.
                # If we tried changing altinh but want to now plot original, deleting the "newIWall" attribute from before.
                try:
                    del self.newIWall
                except AttributeError:
                    pass
            componentNumber += 1
        if disk:
            if verbose:
                print 'CALC_TOTAL: Adding disk component to the total flux.'
            totFlux     = totFlux + self.data['disk']
            componentNumber += 1
        if owall:
            if verbose:
                print 'CALC_TOTAL: Adding outer wall component to the total flux.'
            if altOuter != None:
                self.newOWall = self.data['owall'] * altOuter
                totFlux = totFlux + self.newOWall           # Note: if save=1, will save owall w/ the original altinh.
                self.owallH   = self.altinh * altOuter
            else:
                totFlux       = totFlux + self.data['owall']
                self.owallH   = self.altinh
                # If we tried changing altinh but want to now plot original, deleting the "newOWall" attribute from before.
                try:
                    del self.newOWall
                except AttributeError:
                    pass
            componentNumber += 1
        if dust != 0:
            try:
                dustNum = numCheck(dust, high=dust_high)
            except:
                raise ValueError('CALC_TOTAL: Error! Dust input not a valid integer')
            dustHDU     = fits.open(self.dpath+self.name+'_OTD_'+dustNum+'.fits')
            if verbose:
                print 'CALC_TOTAL: Adding optically thin dust component to total flux.'
            if self.new:
                self.data['dust'] = dustHDU[0].data[1,:]
            else:    
                self.data['dust'] = dustHDU[0].data[:,1]
            totFlux     = totFlux + self.data['dust']
            componentNumber += 1
        
        # If scattered emission is in the dictionary, add it:
        if 'scatt' in self.data.keys():
            scatt       = 1
            if verbose:
                print('CALC_TOTAL: Adding scattered light component to the total flux.')
            totFlux     = totFlux + self.data['scatt']
            componentNumber += 1
        
        # Add the total flux array to the data dictionary attribute:
        if verbose:
            print 'CALC_TOTAL: Total flux calculated. Adding to the data structure.'
        self.data['total'] = totFlux
        componentNumber += 1
        
        # If save, create an output file with these components printed out:
        if save:
            outputTable = np.zeros([len(totFlux), componentNumber])

            # Populate the header and data table with the components and names:
            headerStr   = 'Wavelength, Total Flux, '
            outputTable[:, 0] = self.data['wl']
            outputTable[:, 1] = self.data['total']
            colNum      = 2
            if phot:
                headerStr += 'Photosphere, '
                outputTable[:, colNum] = self.data['phot']
                colNum += 1
            if wall:
                headerStr += 'Inner Wall, '
                outputTable[:, colNum] = self.data['iwall']
                colNum += 1
            if owall:
                headerStr += 'Outer Wall, '
                outputTable[:, colNum] = self.data['owall']
                colNum += 1
            if disk:
                headerStr += 'Outer Disk, '
                outputTable[:, colNum] = self.data['disk']
                colNum += 1
            if dust != 0:
                headerStr += 'Opt. Thin Dust, '
                outputTable[:, colNum] = self.data['dust']
                colNum += 1
            if scatt:
                headerStr += 'Scattered Light, '
                outputTable[:, colNum] = self.data['scatt']
                colNum += 1
            
            # Trim the header and save:
            headerStr  = headerStr[0:-2]
            filestring = '%s%s_%s.dat' % (self.dpath, self.name, numCheck(self.jobn, high=self.high))
            np.savetxt(filestring, outputTable, fmt='%.3e', delimiter=', ', header=headerStr, comments='#')
        
        return

class TTS_Obs(object):
    """
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
        Initializes instances of the class and loads in data to the proper attributes.
        
        INPUTS
        name: The name of the target for which the data represents.
        """
        # Initalize attributes as empty. Can add to the data later.
        self.name       = name
        self.spectra    = {}
        self.photometry = {}
        self.ulim       = []
        
    def add_spectra(self, scope, wlarr, fluxarr, spec_err=None, nod_err=None):
        """
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
                    if spec_err == None and nod_err == None:
                        self.spectra[scope] = {'wl': wlarr, 'lFl': fluxarr}
                    elif spec_err != None and nod_err == None:
                        self.spectra[scope] = {'wl': wlarr, 'lFl': fluxarr, 'specErr': spec_err}
                    elif spec_err == None and nod_err != None:
                        self.spectra[scope] = {'wl': wlarr, 'lFl': fluxarr, 'nodErr': nod_err}
                    elif spec_err != None and nod_err != None:
                        self.spectra[scope] = {'wl': wlarr, 'lFl': fluxarr, 'specErr': spec_err, 'nodErr': nod_err}
                    break
                elif proceed.upper() == 'N' or proceed.upper() == 'NO': # If N or No, do not overwrite data and return
                    print 'ADD_SPECTRA: Will not replace entry. Returning now.'
                    return
                else:
                    tries       = tries + 1                             # If something else, lets you try again
            else:
                raise IOError('You did not enter the correct Y/N response. Returning without replacing.')   # If you enter bad response too many times, raise error.
        else:
            if spec_err == None and nod_err == None:
                self.spectra[scope] = {'wl': wlarr, 'lFl': fluxarr}
            elif spec_err != None and nod_err == None:
                self.spectra[scope] = {'wl': wlarr, 'lFl': fluxarr, 'specErr': spec_err}
            elif spec_err == None and nod_err != None:
                self.spectra[scope] = {'wl': wlarr, 'lFl': fluxarr, 'nodErr': nod_err}
            elif spec_err != None and nod_err != None:
                self.spectra[scope] = {'wl': wlarr, 'lFl': fluxarr, 'specErr': spec_err, 'nodErr': nod_err}
        return
    
    def add_photometry(self, scope, wlarr, fluxarr, errors=None, ulim=0):
        """
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
        Saves the object as a pickle. Damn it Jim, I'm a doctor not a pickle farmer!
        
        WARNING: If you reload the module BEFORE you save the observations as a pickle, this will NOT work! I'm not
        sure how to go about fixing this issue, so just be aware of this.
        
        INPUTS
        picklepath: The path where you will save the pickle. I recommend datapath for simplicity.
        
        OUTPUT:
        A pickle file of the name [self.name]_obs.pkl in the directory provided in picklepath.
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

class Red_Obs(TTS_Obs):
    """
    A similar class to TTS_Obs, except meant to be utilized for observations that have not yet been
    dereddened. Once dereddened, the pickle will be saved as a TTS_Obs object. If saved prior to
    dereddening, the pickle will be associated with Red_Obs instead. I recommend keeping both.
    
    """
    
    def dered(self, Av, Av_unc, law, picklepath, flux=1, lpath=edgepath, err_prop=1):
        """
        Deredden the spectra/photometry present in the object, and then convert to TTS_Obs structure.
        This function is adapted from the IDL procedure 'dered_calc.pro' (written by Melissa McClure).
        
        INPUTS
        Av: The Av extinction value.
        Av_unc: The uncertainty in the Av value provided.
        law: The extinction law to be used -- these extinction laws are found in the ext_laws.pkl file.
             The options you have are 'mkm09_rv5', 'mkm09_rv3', and 'mathis90_rv3.1'
        picklepath: Where your dereddened observations pickle will be saved.
        flux: BOOLEAN -- if True (1), the function will treat your photometry as flux units (erg s-1 cm-2).
              if False (0), the function will treat your photometry as being Flambda (erg s-1 cm-2 cm-1).
        lpath: Where the 'ext_laws.pkl' file is located. I suggest hard coding it as 'edgepath'.
        err_prop: BOOLEAN -- if True (1), will propagate the uncertainty of your photometry with the
                  uncertainty in your Av. Otherwise, it will not.
        
        OUTPUT
        There will be a pickle file called '[self.name]_obs.pkl' in the path provided in picklepath. If
        there is already an obs pickle file there, it will add an integer to the name to differentiate
        between the two files, rather than overwriting.
        """
        
        # Read in the dereddening laws pickle. The default is whereever you keep EDGE.py, but you can move it.
        extPickle = open(lpath + 'ext_laws.pkl', 'rb')
        extLaws   = cPickle.load(extPickle)
        extPickle.close()
        
        # Figure out which law we will be using based on the user input and Av:
        if law == 'mkm09_rv5':
            print('Using the McClure (2009) ext. laws for Av >3\nwith the Mathis (1990) Rv=5.0 \
                   law for Av < 3\n(appropriate for molecular clouds like Ophiuchus).')
            AjoAks = 2.5341
            AvoAj  = 3.06
            
            if Av >= 8.0:                                       # high Av range
                wave_law = extLaws['mkm_high']['wl']
                ext_law  = extLaws['mkm_high']['ext'] / AjoAks
            elif Av >= 3.0 and Av < 8.0:                        # med Av range
                wave_law = extLaws['mkm_med']['wl']
                ext_law  = extLaws['mkm_med']['ext'] / AjoAks
            elif Av < 3.0:                                      # low Av range
                wave_law = extLaws['mathis_rv5']['wl']
                ext_law  = extLaws['mathis_rv5']['ext']
            else:
                raise ValueError('DERED: Specified Av is not within acceptable ranges.')
        elif law == 'mkm09_rv3':
            print('Using the McClure (2009) ext. laws for Av >3\nwith the Mathis (1990) Rv=3.1 \
                   law for Av < 3\n(appropriate for molecular clouds like Taurus).')
            AjoAks = 2.5341
            AvoAj  = 3.55 # for Rv=3.1 **WARNING** this is still the Rv=5 curve until 2nd step below.
            
            if Av >= 8.0:                                       # high Av range
                wave_law = extLaws['mkm_high']['wl']
                ext_law  = extLaws['mkm_high']['ext'] / AjoAks
            elif Av >= 3.0 and Av < 8.0:                        # med Av range
                wave_law = extLaws['mkm_med']['wl']
                ext_law  = extLaws['mkm_med']['ext'] / AjoAks
            elif Av < 3.0:                                      # low Av range
                wave_law = extLaws['mathis_rv3']['wl']
                ext_law  = extLaws['mathis_rv3']['ext']
            else:
                raise ValueError('DERED: Specified Av is not within acceptable ranges.')
            
            # Fix to the wave and ext. law:
            wave_jm  = extLaws['mathis_rv3']['wl']
            ext_jm   = extLaws['mathis_rv3']['ext']
            jindmkm  = np.where(wave_law >= 1.25)[0]
            jindjm   = np.where(wave_jm < 1.25)[0]
            wave_law = np.append(wave_jm[jindjm], wave_law[jindmkm])
            ext_law  = np.append(ext_jm[jindjm], ext_law[jindmkm])
            
        elif law == 'mathis90_rv3.1':
            print('Using the Mathis (1990) Rv=3.1 law\n(appropriate for diffuse ISM).')
            AjoAks   = 2.5341
            AvoAj    = 3.55                                     # for Rv=3.1
            wave_law = extLaws['mathis_rv3']['wl']
            ext_law  = extLaws['mathis_rv3']['ext']
        else:
            raise ValueError('DERED: Specified extinction law string is not recognized.')
        
        A_object        = Av
        A_object_string = str(round(A_object,2))
        
        # Open up a new TTS_Obs object to take the dereddened values:
        deredObs        = TTS_Obs(self.name)
        
        # Loop over the provided spectra (and possible errors), and compute the dereddened fluxes
        # and uncertainties where available. The possible uncertainties calculations are both the
        # SSC/SMART's "spectral uncertainty" and the "nod-differenced uncertainty".
        for specKey in self.spectra.keys():
            extInterpolated = np.interp(self.spectra[specKey]['wl'], wave_law, ext_law) # Interpolated ext.
            A_lambda        = extInterpolated * (A_object / AvoAj)
            spec_flux       = np.float64(self.spectra[specKey]['lFl']*10.0**(0.4*A_lambda))

            if 'specErr' in self.spectra[specKey].keys():
                spec_unc    = np.float64(spec_flux*np.sqrt(((self.spectra[specKey]['specErr']/self.spectra[specKey]['lFl'])\
                                         **2.) + (((0.4*math.log(10)*extInterpolated*Av_unc)/(AvoAj))**2.)) )
            else:
                spec_unc    = None
            if 'nodErr' in self.spectra[specKey].keys():
                pn_vals     = self.spectra[specKey]['nodErr']/abs(self.spectra[specKey]['nodErr'])
                spec_nod    = np.float64(pn_vals*spec_flux*np.sqrt(((self.spectra[specKey]['nodErr'] \
                                         / self.spectra[specKey]['lFl'])**2.) + \
                                         (((0.4*math.log(10)*extInterpolated*Av_unc)/(AvoAj))**2.)) )
            else:
                spec_nod    = None
            # Correct units to flux:
            spec_flux       = spec_flux * self.spectra[specKey]['wl'] * 1e-4
            if spec_unc != None:
                spec_unc    = spec_unc  * self.spectra[specKey]['wl'] * 1e-4
            if spec_nod != None:
                spec_nod    = spec_nod  * self.spectra[specKey]['wl'] * 1e-4
            deredObs.add_spectra(specKey, self.spectra[specKey]['wl'], spec_flux, spec_err=spec_unc, nod_err=spec_nod)
        
        # Spectra are done, onwards to photometry:
        for photKey in self.photometry.keys():
            extInterpolated = np.interp(self.photometry[photKey]['wl'], wave_law, ext_law)
            A_lambda        = extInterpolated * (A_object / AvoAj)
            if flux:
                photcorr    = self.photometry[photKey]['lFl'] / (self.photometry[photKey]['wl']*1e-4)
            else:
                photcorr    = self.photometry[photKey]['lFl']
            phot_dered      = np.float64(photcorr*10.0**(0.4*A_lambda))
            if 'err' in self.photometry[photKey].keys():
                if flux:
                    errcorr = self.photometry[photKey]['err'] / (self.photometry[photKey]['wl']*1e-4)
                else:
                    errcorr = self.photometry[photKey]['err']
                if err_prop:
                    phot_err= np.float64(photcorr * np.sqrt(((errcorr/photcorr)**2.) + \
                                         (((0.4*math.log(10.)*extInterpolated*Av_unc)/AvoAj)**2.)) )
                else:
                    phot_err= np.float64(errcorr*10.0**(0.4*A_lambda)) # Without propogating error!
            else:
                phot_err    = None
            if photKey in self.ulim:
                ulimVal     = 1
            else:
                ulimVal     = 0
            # Now, convert everything to flux units:
            phot_dered      = phot_dered * self.photometry[photKey]['wl'] * 1e-4
            phot_err        = phot_err * self.photometry[photKey]['wl'] * 1e-4
            deredObs.add_photometry(photKey, self.photometry[photKey]['wl'], phot_dered, errors=phot_err, ulim=ulimVal)

        # Now that the new TTS_Obs object has been created and filled in, we must save it:
        deredObs.SPPickle(picklepath=picklepath)
        
        return
    
    def SPPickle(self, picklepath):
        """
        The new version of SPPickle, different so you can differentiate between red and dered pickles.
        
        INPUT
        picklepath: The path where the new pickle file will be located.
        
        OUTPUT
        A new pickle file in picklepath, of the name [self.name]_red.pkl
        """
        
        # Check whether or not the pickle already exists:
        pathlist        = filelist(picklepath)
        outname         = self.name + '_red.pkl'
        count           = 1
        while 1:
            if outname in pathlist:
                if count == 1:
                    print 'SPPICKLE: Pickle already exists in directory. For safety, will change name.'
                countstr= numCheck(count)
                count   = count + 1
                outname = self.name + '_red_' + countstr + '.pkl'
            else:
                break
        # Now that that's settled, let's save the pickle.
        f               = open(picklepath + outname, 'wb')
        cPickle.dump(self, f)
        f.close()
        return
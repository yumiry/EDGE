import numpy as np
from astropy.io import fits
from astropy.io import ascii
from glob import glob
#import pdb

def collate(path, jobnum, name, destination, optthin=0, clob=0):
    """
     collate.py                                                                          
                                                                                           
     PURPOSE:                                                                              
            Organizes and stores flux and parameters from the D'Alessio                    
            disk/optically thin dust models and jobfiles in a fits                         
            file with a header                                                             
                                                                                           
     CALLING SEQUENCE:                                                                     
            collate(path, jobnum, name, destination, [optthin=1], [clob=1])
                                                                                           
                                                                                           
     INPUTS:                                                                               
            path: String of with path to location of jobfiles and model result             
                  files. Both MUST be in the same location!                                      
                                                                                       
            jobnum: String associated with a job number label end.                         
                                                                                            
            name: String of the name of the object                                         
                                                                                            
            Destination: String with where you want the fits file to be                    
                         sent after it's made                                                           
                                                                                            
     KEYWORDS                                                                              
            optthin: Set this value to 1 (or True) to run the optically thin dust
                     version of collate instead of the normal disk code. This will
                     also place a tag in the header.
             
            clob: Set this value to 1 (or True) to overwrite a currently existing
                  fits file from a previous run.

     THESE OPTIONS NOT CURRENTLY SUPPORTED IN PYTHON VERSION OF CODE                                                                                  
            /innerdisk: Denotes that this is an inner disk for a                           
            pre-transitional disk. This doesn't change what code runs, just                
            adds a tag to the header.                                                      
                                                                                            
            /outerdisk: Denotes that this is an outer disk for a                           
            pre-transitional disk. This doesn't change what code runs, just                
            adds a tag to the header.                                                      
                                                                                            
            /nounderscore: if the models are saved without an underscore                   
            in the objectname, use this flag. Eg:                                          
            rin.t130.amax3p0.test004 vs. rin.t130.amax3p0.test_004                         
                                                                                            
            /nophotnum: if the photosphere does not have a model                           
            associated with it, use this flag. Eg: Phot4350.test                           
                                                                                            
     NOTES:                                                                                
            ****CURRENTLY CANNOT TAKE IN VECTORS AND MUST BE USED IN A FOR                 
            LOOP DUE TO USE OF READCOL****                                                 
                                                                                            
            Label ends for model results should of form objectname_001,                    
            objectname_002, ...                                                            
                                                                                            
            For disk models, job file name convention is job001, job002, ...               
                                                                                            
            For optically thin dust, job file name convention is                           
            job_optthin001, job_optthin002, ...                                            

            !!!!!!! AMAX IN OPTTHIN MODEL DID NOT ORIGINALLY HAVE AN 'S' AFTER IT. CHANGED
            IN PYTHON VERSION TO AMAXS

                                                                                            
     MODIFICATION HISTORY
     Dan Feldman, 25 June 2015, Improved readability.                                      
     Connor Robinson, Dan Feldman, 24 June 2015, Finished all current functionality for use
     Connor Robinson 26 May 2015, Began work on optically thin disk code
     Connor Robinson, Dan Feldman, 22 May 2015, Wrote disk code in python
     Connor Robinson 3, Mar, 2015, Added the /nounderscore and /photnum flags              
     Connor Robinson 6 Nov, 2014 First version uploaded to cluster  
                                                                                            
     -                                                                                      
    """
    
    # If working with optically thin models
    if optthin:
        
        #Read in file
        #NOTE: NEED TO CHANGE TO ADD UNDERSCORE AS DEFAULT JOB NAME
        job = 'job_optthin'+jobnum
        f = open(path+job, 'r')
        jobf  = f.read()
        f.close()
        
        
        #Define what variables to record
        sdparam = (['TSTAR', 'RSTAR', 'DISTANCIA', 'MUI', 'ROUT', 'RIN', 'TAUMIN', 'POWER',
                    'FUDGEORG', 'FUDGETROI', 'FRACSIL', 'FRACENT', 'FRACFORST', 'FRACAMC', 
                    'AMAXS'])
        dparam = np.zeros(len(sdparam), dtype = float)
        
        #Read in the data associated with this model
        data = ascii.read(glob(path+'fort16*'+jobnum)[0])
        
        #Combine data into a single array to be consistant with previous version of collate
        dataarr = np.array([data['col1'], data['col3']])
        
        #Make an HDU object to contain header/data
        hdu = fits.PrimaryHDU(dataarr)
        
        #Parse variables according to convention in job file
        for ind, param in enumerate(sdparam):
            
            #Handles the case of AMAXS which is formatted slightly differently
            if param  == 'AMAXS':
                for num in range(10):
                    if jobf.split("lamax='amax")[num].split("\n")[-1][0] == 's':
                        samax = jobf.split("lamax='amax")[num+1].split("'")[0]
                        if samax == '1mm':
                            hdu.header.set(param, 1000.)
                        else:
                            hdu.header.set(param, float(samax.replace('p', '.')))
                            
                            
            #Handle the rest of the variables
            else:
                paramold = param
                if param   == 'DISTANCIA':
                    param = 'DISTANCE' #Reduce the amount of Spanish here
                elif param == 'FUDGETROI':
                    param = 'FUDGETRO'
                elif param == 'FRACFORST':
                    param = 'FRACFORS'
                hdu.header.set(param, float(jobf.split("set "+paramold+"='")[1].split("'")[0]))
                
        hdu.header.set('OBJNAME', name)
        hdu.header.set('JOBNUM', jobnum)
        hdu.header.set('OPTTHIN', 1)
        hdu.header.set('WLAXIS', 0)
        hdu.header.set('LFLAXIS',1)
        hdu.writeto(destination+name+'_OTD_'+jobnum+'.fits', clobber = clob)
        
    # If working with job models, not optically thin models
    elif optthin == 0 or optthin == 'False':
        
        #read in file
        job = 'job'+jobnum
        f = open(path+job, 'r')
        jobf = f.read()
        f.close()
        
        #Define what variables to record
        sparam = (['MSTAR', 'TSTAR', 'RSTAR', 'DISTANCIA','MDOT', 'ALPHA', 'MUI', 'RDISK',
                   'AMAXS', 'EPS', 'WLCUT_ANGLE', 'WLCUT_SCATT', 'NSILCOMPOUNDS', 'SILTOTABUN',
                   'AMORPFRAC_OLIVINE', 'AMORPFRAC_PYROXENE', 'FORSTERITE_FRAC', 'ENSTATITE_FRAC', 
                   'TEMP', 'ALTINH', 'TSHOCK'])
        dparam = np.zeros(len(sparam), dtype = float)
        
        #Parse variables according to convention in the job file
        for ind, param in enumerate(sparam):
            if param == 'AMAXS':
                num_amax = 10 #Number of choices for AMAX, including the case where amax can be 1mm (1000 microns)
                for num in range(num_amax):
                    if jobf.split("AMAXS='")[num+1].split("\n")[1][0] == '#':
                        continue
                    elif jobf.split("AMAXS='")[num+1].split("\n")[1][0] == 's':
                        dparam[ind] = float(jobf.split(param+"='")[num+1].split("'")[0])
                    elif dparam[ind] == 0. and num == num_amax-1:
                        dparam[ind] = 1000. #HANDLES THE CASE THAT MM SIZED DUST GRAINS EXIST IN JOBFILE
            
            elif param == 'EPS':
                for num in range(7):
                    if jobf.split("EPS='")[num+1].split("\n")[1][0] == '#' and num != 7:
                        continue
                    elif jobf.split("EPS='")[num+1].split("\n")[1][0] == 's':
                        dparam[ind] = float(jobf.split(param+"='")[num+1].split("'")[0])
                    else: 
                        raise IOError('COLLATE FAILED ON EPSILON VALUE. FIX JOB FILE '+jobnum)
            
            elif param == 'TEMP' or param == 'TSHOCK':
                try:
                    dparam[ind] = float(jobf.split(param+"=")[1].split(".")[0])
                except ValueError:
                    raise ValueError('COLLATE: MISSING . AFTER '+param+' VALUE, GO FIX IN JOB FILE ' +jobnum)
            
            elif param == 'ALTINH':
                try:
                    dparam[ind] = float(jobf.split(param+"=")[1].split(" ")[0])
                except ValueError:
                    raise ValueError('COLLATE MISSING SPACE [ ] AFTER ALTINH VALUE, GO FIX IN JOB FILE'+jobnum)
            
            else:
                dparam[ind] = float(jobf.split(param+"='")[1].split("'")[0])
        
        #Rename header labels that are too long
        sparam[sparam.index('AMORPFRAC_OLIVINE')]  = 'AMORF_OL'
        sparam[sparam.index('AMORPFRAC_PYROXENE')] = 'AMORF_PY'
        sparam[sparam.index('WLCUT_ANGLE')] = 'WLCUT_AN'
        sparam[sparam.index('WLCUT_SCATT')] = 'WLCUT_SC'
        sparam[sparam.index('NSILCOMPOUNDS')] = 'NSILCOMP'
        sparam[sparam.index('SILTOTABUN')] = 'SILTOTAB'
        sparam[sparam.index('FORSTERITE_FRAC')] = 'FORSTERI'
        sparam[sparam.index('ENSTATITE_FRAC')] = 'ENSTATIT'
        
        #Reduce the amount of Spanish here
        sparam[sparam.index('DISTANCIA')] = 'DISTANCE'

        #Read in data from outputs
        phot  = ascii.read(glob(path+'Phot*'+jobnum)[0]) #if you don't change photospheres for each run, may need to change.
        angle = ascii.read(glob(path+'angle*'+'_'+jobnum+'*')[0], data_start = 1)
        wall  =  ascii.read(glob(path+'fort17*'+name+'_'+jobnum)[0], data_start = 9)
        
        dataarr = np.array([phot['col1'], phot['col2'], wall['col2'], angle['col4']])
        
        hdu = fits.PrimaryHDU(dataarr)
        
        for i, param in enumerate(sparam):
            hdu.header.set(param, dparam[i])
        
        #Add in other stuff to header
        hdu.header.set('RIN', float(np.loadtxt(glob(path+'rin*'+name+'_'+jobnum)[0])))
        hdu.header.set('WLAXIS', 0)
        hdu.header.set('PHOTAXIS',1)
        hdu.header.set('WALLAXIS', 2)
        hdu.header.set('ANGAXIS', 3)
        
        #Write header to fits file
        hdu.writeto(destination+name+'_'+jobnum+'.fits', clobber = clob)
        
    # If you don't give a valid input for the optthin keyword
    else:
        raise IOError('COLLATE: INVALID INPUT FOR OPTTHIN KEYWORD, SHOULD BE 1 OR 0')
    
    return

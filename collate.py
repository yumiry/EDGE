import numpy as np
from astropy.io import fits
from astropy.io import ascii
from glob import glob
#import pdb

def collate(path, jobnum, name, destination, optthin=0, clob=0, high=0, noextinct = 0, noangle = 0, nowall = 0, nophot = 0, noscatt = 1):
    """
     collate.py                                                                          
                                                                                           
     PURPOSE:                                                                              
            Organizes and stores flux and parameters from the D'Alessio                    
            disk/optically thin dust models and jobfiles in a fits                         
            file with a header.                                                             
                                                                                           
     CALLING SEQUENCE:                                                                     
            collate(path, jobnum, name, destination, [optthin=1], [clob=1], [noextinc = 1])
                                                                                           
                                                                                           
     INPUTS:                                                                               
            path: String of with path to location of jobfiles and model result             
                  files. Both MUST be in the same location!                                      
                                                                                       
            jobnum: String or integer associated with a job number label end.                         
                                                                                            
            name: String of the name of the object                                         
                                                                                            
            Destination: String with where you want the fits file to be                    
                         sent after it's made                                                           
                                                                                            
     OPTIONAL KEYWORDS                                                                             
            optthin: Set this value to 1 (or True) to run the optically thin dust
                     version of collate instead of the normal disk code. This will
                     also place a tag in the header.
             
            clob: Set this value to 1 (or True) to overwrite a currently existing
                  fits file from a previous run.
            
            high: Set this value to 1 (or True) if your job number is 4 digits long.

            noextin: Set this value to 1 (or True) if you do NOT want to apply extinction 
                     to the inner wall and photosphere.

            noscatt: !!!!! NOTE: THIS IS SET TO 1 BY DEFAULT !!!!!
                     Set this value to 1 (or True) if you do NOT want to include the scattered light file.
                     Set this value to 0 (or False) if you DO want to include the scattered light file

     EXAMPLES:                                                                                
            To collate a single model run for the object 'myobject' under the
            job number '001', use the following commands:

            from collate import collate
            path = 'Some/path/on/the/cluster/where/your/model/file/is/located/'
            name = 'myobject'
            dest = 'where/I/want/my/collated/file/to/go/'
            modelnum = 1 
            collate(path, modelnum, name, dest) 
   
            Note that:
            modelnum = '001' will also work.

            collate.py cannot handle multiple models at once, and currently needs to be
            run in a loop. An example run with 100 optically thin dust models would
            look something like this:

            from collate import collate
            path = 'Some/path/on/the/cluster/where/your/model/files/are/located/'
            name = 'myobject'
            dest = 'where/I/want/my/collated/files/to/go/'
            for i in range(100):
                collate(path, i+1, name, dest, optthin = 1)
                                    
               
     NOTES:
    
            For the most current version of collate and EDGE, please visit the github respository:
            https://github.com/danfeldman90/EDGE

            Collate corrects the flux from the star and the inner wall for extinction from
            the outer disk.

            Label ends for model results should of form objectname_001,                    
                                                                                            
            For disk models, job file name convention is job001
                                                                                            
            For optically thin dust, job file name convention is job_optthin001

            amax in the optthin model did not originally have an s after it. It is changed in 
            the header file to have the s to be consistant with the disk models.



    

                                                                                            
     MODIFICATION HISTORY
     Connor Robinson, 30 July 2015, Added scattered light + ability to turn off components of the model
     Connor Robinson, 24 July 2015, Added extinction from the outer disk  + flag to turn it off
     Connor Robinson, 23 July 2015, Updated documentation and added usage examples
     Dan Feldman, 19 July 2015, added numCheck() and high kwarg to handle integer jobnums
     Dan Feldman, 25 June 2015, Improved readability.                                      
     Connor Robinson, Dan Feldman, 24 June 2015, Finished all current functionality for use
     Connor Robinson 26 May 2015, Began work on optically thin disk code
     Connor Robinson, Dan Feldman, 22 May 2015, Wrote disk code in python
     Connor Robinson 3, Mar, 2015, Added the /nounderscore and /photnum flags              
     Connor Robinson 6 Nov, 2014 First version uploaded to cluster  
                                                                                            
    """
    
    # Convert jobnum into a string:
    if type(jobnum) == int:
        jobnum = numCheck(jobnum, high=high)
        
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
                    raise ValueError('COLLATE MISSING SPACE [ ] AFTER ALTINH VALUE, GO FIX IN JOB FILE '+jobnum)
            
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

        #Read in data from outputs (if the no____ flags are not set)
        
        #set up empty array to accept data, column names and axis number
        dataarr = np.array([])
        axis = {'WLAXIS':0}
        axis_count = 1 #Starts at 1, axis 0 reserved for wavelength information

        #Read in arrays and manage axis information

        if nophot == 0:
            phot  = ascii.read(glob(path+'Phot*'+jobnum)[0]) 
            axis['PHOTAXIS'] = axis_count

            dataarr = np.concatenate((dataarr, phot['col1']))
            dataarr = np.concatenate((dataarr, phot['col2']))

            axis_count += 1

        elif nophot != 1 and nophot != 0:
            raise IOError('COLLATE: INVALID INPUT FOR NOPHOT KEYWORD, SHOULD BE 1 OR 0')

        if nowall == 0:
            wall  =  ascii.read(glob(path+'fort17*'+name+'_'+jobnum)[0], data_start = 9)
            axis['WALLAXIS'] = axis_count

            #If the photosphere was not run, then grab wavelength information from wall file
            if nophot != 0: 
                dataarr = np.concatenate((dataarr, wall['col1']))
            
            dataarr = np.concatenate((dataarr, wall['col2']))
            axis_count += 1

        elif nowall != 1 and nowall != 0:
            raise IOError('COLLATE: INVALID INPUT FOR NOWALL KEYWORD, SHOULD BE 1 OR 0')

        if noangle == 0:
            angle = ascii.read(glob(path+'angle*'+name+'_'+jobnum+'*')[0], data_start = 1)
            axis['ANGAXIS'] = axis_count

            #If the photosphere was not run, and the wall was not run then grab wavelength information from angle file
            if nophot != 0 and nowall != 0:
                dataarr = np.concatenate((dataarr, angle['col1']))

            dataarr = np.concatenate((dataarr, angle['col4']))
            axis_count += 1

        elif noangle != 1 and noangle != 0:
            raise IOError('COLLATE: INVALID INPUT FOR NOANGLE KEYWORD, SHOULD BE 1 OR 0')

        if noscatt == 0:
            scatt = ascii.read(glob(path+'scatt*'+name+'_'+jobnum+'*')[0], data_start = 1)
            axis['SCATAXIS'] = axis_count
            
            if nophot != 0 and nowall != 0 and noangle != 0:
                dataarr = np.concatenate((dataarr, scatt['col1']))
            
            dataarr = np.concatenate((dataarr, scatt['col4']))
            axis_count += 1

        elif noscatt != 1 and noscatt != 0:
            raise IOError('COLLATE: INVALID INPUT FOR NOSCATT KEYWORD, SHOULD BE 1 OR 0')

        if noextinct == 0:
            if noangle != 0:
                raise IOError('NEED A ANGLE FILE IN ORDER TO INCLUDE EXTINCTION FROM DISK, NOANGLE KEYWORD SHOULD BE UNDEFINED OR 0')

            dataarr = np.concatenate((dataarr, angle['col6']))
            axis['EXTAXIS'] = axis_count


            axis_count += 1

        elif noextinct != 1 and noextinct != 0:
            raise IOError('COLLATE: INVALID INPUT FOR NOANGLE KEYWORD, SHOULD BE 1 OR 0')


        #Put data array into the standard form for EDGE
        dataarr = np.reshape(dataarr, (axis_count, len(dataarr)/axis_count))

        if noextinct == 0:
            if nophot == 0:
                dataarr[axis['PHOTAXIS'],:] *=np.exp((-1)*dataarr[axis['EXTAXIS'],:])
            if nowall == 0:
                dataarr[axis['WALLAXIS'],:] *=np.exp((-1)*dataarr[axis['EXTAXIS'],:])
            


        #Create the header and add parameters
        hdu = fits.PrimaryHDU(dataarr)
        
        for i, param in enumerate(sparam):
            hdu.header.set(param, dparam[i])

        hdu.header.set('RIN', float(np.loadtxt(glob(path+'rin*'+name+'_'+jobnum)[0])))
        
        #Create tags in the header that match up each column to the data enclosed]
        for naxis in axis:
            hdu.header.set(naxis, axis[naxis])

        #Add a tag to the header if the noextinct flag is on
        if noextinct == 1:
            hdu.header.set('NOEXT', 1)
        
        #Write header to fits file
        hdu.writeto(destination+name+'_'+jobnum+'.fits', clobber = clob)
        

    # If you don't give a valid input for the optthin keyword, raise an error
    else:
        raise IOError('COLLATE: INVALID INPUT FOR OPTTHIN KEYWORD, SHOULD BE 1 OR 0')
    
    return

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

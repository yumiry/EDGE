#!/usr/bin/env python

# If you have any observations pickles from the older version of EDGE (Model_CodeV2),
# you can use this script to convert them to new pickles in EDGE.
# Replace the old and new paths with where the old and new pickles will be, respectively.
# Replace the 'objectname' in loadPickle and TTS_Obs with the name you used for our
# nameing convention in the pickle (the string before the '_001' part of the filename).
# If you need to adapt this or have questions, email Dan (danfeldman90@gmail.com)

import Model_CodeV2 as mc2
import EDGE as edge

# Set the path where the old pickle is, and where the new one will be saved:
oldPath = '/Users/etc./'
newPath = '/Users/etc./'

# Load in the old pickle from mc2:
old_pkl = mc2.loadPickle('objectname', picklepath=oldPath)

# Create a new pickle in the new edge format:
new_pkl = edge.TTS_Obs('objectname')
for specName in old_pkl.spectra.keys():
    try:
        new_pkl.add_spectra(specName, old_pkl.spectra[specName]['wl'], old_pkl.spectra[specName]['lFl'], \
                            old_pkl.spectra[specName]['err'])
    except KeyError:
        new_pkl.add_spectra(specName, old_pkl.spectra[specName]['wl'], old_pkl.spectra[specName]['lFl'])
for photName in old_pkl.photometry.keys():
    try:
        new_pkl.add_photometry(photName, old_pkl.photometry[photName]['wl'], old_pkl.photometry[photName]['lFl'], \
                               old_pkl.photometry[photName]['err'])
    except KeyError:
        new_pkl.add_photometry(photName, old_pkl.photometry[photName]['wl'], old_pkl.photometry[photName]['lFl'])
new_pkl.ulim = old_pkl.ulim

# At this point, all of the data should have been converted over. So we save the new pickle:
new_pkl.SPPickle(newPath)
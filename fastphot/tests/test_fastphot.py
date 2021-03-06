from __future__ import absolute_import, division, print_function

import numpy as npy
import numpy.testing as npt
import fastphot as FP

def test_gaussian_PSF():
	# """
	# Testing the normalization of the gaussian PSF model
	# For npix_x >> std
    # sum(MAP_PSF) --> 1.0
    
    # """
    # Define PSF properties
    std = 2.0
    npix = int(100. * std)
    # 
    # Create MAP_PSF
    PSF_MAP = FP.gaussian_PSF(npix=npix, std=std)
    #
    # Test
    npt.assert_almost_equal(npy.sum(PSF_MAP), 1.e0, decimal=6)
    
def test_no_noise():
	# """
    # Testing the flux measurement on a small MAP npix_x, npix_y = 50, 50
    # with 11 sources without noise
    # source fluxes are homogeneously distributed between 0.5 and 10.0
    # """
    #
	# create the PSF_MAP
	PSF_MAP = FP.gaussian_PSF(npix=31, std=2.0)
	#
	# Create the SC_MAP with some sources
	npix_x = 50
	npix_y = 50
	N_srcs = 11
	#
	SC_MAP = npy.zeros([npix_x, npix_y])
	MASK_MAP = (SC_MAP > 0.e0)
	# 
	# Define flux domain
	minflux = 5.e-1
	maxflux = 1.e1
	#
	# Build the reference catalog
	for i in range(N_srcs):
		#
		# create the new source
		# ID, x_pos, dx_pos, y_pos, dy_pos, flux, dflux
		xpos = npy.random.uniform(1, npix_x - 2)
		ypos = npy.random.uniform(1, npix_y - 2)
		flux = npy.random.uniform(minflux, maxflux)
		s = (i, xpos, 0.e0, ypos, 0.e0, flux, 0.e0)
		#
		# Update source dict
		if (i == 0):
			# create the structured list
			Catalog = npy.array(s, dtype=FP.src_dtype())
		else:
			Catalog = npy.append(Catalog, npy.array(s, dtype=FP.src_dtype()))	
	#
	SC_MAP = npy.ma.array(FP.model_MAP(SC_MAP, PSF_MAP, Catalog), mask=MASK_MAP)
	# 
	# Create the NOISE_MAP
	NOISE_MAP = npy.ma.array(npy.ones([npix_x, npix_y]), mask=MASK_MAP)
	# 	
	# Mask negative flux sources
	mask = (Catalog['flux'] < 0.)
	Masked_Catalog = npy.ma.array(Catalog, dtype=FP.src_dtype(), mask=mask)	
	# Save initial fluxes
	flux_in = npy.ma.compressed(Masked_Catalog['flux'])
	#
	Final_Catalog, bkg, RESIDUAL_MAP = FP.fastphot(SC_MAP, PSF_MAP, NOISE_MAP, Masked_Catalog, nb_process=4)
	#
	# Test 
	npt.assert_almost_equal(npy.ma.compressed(Final_Catalog['flux']), flux_in, decimal=-6)
	#
	# Save SC_MAP
	FP.save_pdf_MAP(SC_MAP, map_name='SC', src_cat=Final_Catalog)
	# Save RESIDUAL_MAP
	FP.save_pdf_MAP(RESIDUAL_MAP, map_name='RESIDUAL', src_cat=Final_Catalog)

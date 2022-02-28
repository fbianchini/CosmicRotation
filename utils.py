import numpy as np
import os
import numba

arcmin2rad = np.pi / 180. / 60. 
rad2arcmin = 1./arcmin2rad

@numba.jit
def ltoi(l, n, dl):
    return ( int(np.round(l[0]/dl) + n/2) , int(np.round(l[1]/dl) + n/2))

@numba.jit
def itol(i, n, dl):
    return ( dl * (i[0] - n/2), dl * (i[1] - n/2) )

@numba.jit
def dotprod(l1, l2):
    return l1[0]*l2[0] + l1[1]*l2[1]

@numba.jit
def ltoangle(l1):
    angle = np.arctan2(l1[1], l1[0])

    if (angle < 0): angle += 2 * np.pi #// this normalises the angle between [0,2pi)...do we need it?

    return angle

@numba.jit
def anglebetween(l1, l2):
    angle = np.arctan2(l2[1], l2[0]) - np.arctan2(l1[1], l1[0])

    if (angle < 0): angle += 2 * np.pi #// this normalises the angle between [0,2pi)...do we need it?

    return angle

@numba.jit
def circlecheck(l, lmax, lmin):
    return ( (np.hypot(l[0], l[1]) < lmax) and  (np.hypot(l[0], l[1]) > lmin))

def bl(fwhm_arcmin, lmax=3000):
	""" 
	Returns the map-level transfer function for a symmetric Gaussian beam.
	Parameters
	----------
	fwhm_arcmin : float
		Beam full-width-at-half-maximum (fwhm) in arcmin.
	lmax : int
		Maximum multipole.
	Returns
	-------
	bl : array
		Gaussian beam function
	"""
	ls = np.arange(0, lmax+1)
	return np.exp( -ls*(ls+1.) * (fwhm_arcmin * np.pi/180./60.)**2 / (16.*np.log(2.)) )

def nl_cmb(noise_uK_arcmin, fwhm_arcmin, lmax=3000, lknee=None, alpha=None):
	""" 
	Returns the beam-deconvolved noise power spectrum in units of uK^2 for
	Parameters
	----------
	noise_uK_arcmin : float or list  
		Map noise level in uK-arcmin 
	fwhm_arcmin : float or list
		Beam full-width-at-half-maximum (fwhm) in arcmin, must be same size as noise_uK_arcmin
	lmax : int
		Maximum multipole.
	"""
	ls = np.arange(0, lmax+1)
	if np.isscalar(noise_uK_arcmin) or (np.size(noise_uK_arcmin) == 1):
		if (lknee is not None) and (alpha is not None):
			return  ((noise_uK_arcmin * np.pi/180./60.)**2 / bl(fwhm_arcmin, lmax=lmax)**2) * (1. + (lknee/ls)**alpha)
		else:   
			return  ((noise_uK_arcmin * np.pi/180./60.)**2 / bl(fwhm_arcmin, lmax=lmax)**2)
	else:
		return 1./np.sum([1./nl_cmb(noise_uK_arcmin[i], fwhm_arcmin[i], lmax=lmax, lknee=lknee, alpha=alpha) for i in xrange(len(noise_uK_arcmin))], axis=0)

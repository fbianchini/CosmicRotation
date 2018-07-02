import numpy as np
import numba
from utils import *
import matplotlib.pyplot as plt 

# ===========================
# ====== FILTERS ============
# ===========================

@numba.jit
def smallf_EB(l1, l2, eepowerspec2d, bbpowerspec2d, n, dl):
	# this is f_EB for rotation, a.k.a W^{alpha,EB}
	# eepowerspec2d and bbpowerspec2d are 2d arrays containing the signal only EE and BB spectra

	i1 = ltoi(l1, n, dl)
	i2 = ltoi(l2, n, dl)

	bigell_x = l1[0] + l2[0]
	bigell_y = l1[1] + l2[1]
	phi_l1l2 = anglebetween(l1, l2)

	output = 2.* (eepowerspec2d[i1[0]][i1[1]] - bbpowerspec2d[i2[0]][i2[1]]) * np.cos(2.0 * phi_l1l2)

	return output
	
@numba.jit
def filter_EB(l1, l2, eepowerspec2d, bbpowerspec2d, eetotallenspowerspec2d, bbtotallenspowerspec2d,  n,  dl ):

	i1 = ltoi(l1, n, dl)
	i2 = ltoi(l2, n, dl)

	num = smallf_EB(l1, l2, eepowerspec2d, bbpowerspec2d, n, dl )

	if (eetotallenspowerspec2d[i1[0]][i1[1]] == 0.): return 0.
	if (eetotallenspowerspec2d[i2[0]][i2[1]] == 0.): return 0.
	if (bbtotallenspowerspec2d[i1[0]][i1[1]] == 0.): return 0.
	if (bbtotallenspowerspec2d[i2[0]][i2[1]] == 0.): return 0.
  
	denom = eetotallenspowerspec2d[i1[0]][i1[1]] * bbtotallenspowerspec2d[i2[0]][i2[1]]
  
	return num / denom

@numba.jit
def smallf_TB(l1, l2, tepowerspec2d,  n,  dl ):

	i1 = ltoi(l1, n, dl)
	i2 = ltoi(l2, n, dl)

	bigell_x = l1[0] + l2[0]
	bigell_y = l1[1] + l2[1]
	phi_l1l2 = anglebetween(l1, l2)

	output = 2.* tepowerspec2d[i1[0]][i1[1]] * np.cos(2.0 * phi_l1l2)

	return output
	
@numba.jit
def filter_TB(l1, l2, tepowerspec2d, tttotallenspowerspec2d, bbtotallenspowerspec2d,  n,  dl ):

	i1 = ltoi(l1, n, dl)
	i2 = ltoi(l2, n, dl)

	num = smallf_TB(l1, l2, tepowerspec2d, n, dl )

	if (tttotallenspowerspec2d[i1[0]][i1[1]] == 0.): return 0.
	if (tttotallenspowerspec2d[i2[0]][i2[1]] == 0.): return 0.
	if (bbtotallenspowerspec2d[i1[0]][i1[1]] == 0.): return 0.
	if (bbtotallenspowerspec2d[i2[0]][i2[1]] == 0.): return 0.
  
	denom = tttotallenspowerspec2d[i1[0]][i1[1]] * bbtotallenspowerspec2d[i2[0]][i2[1]]
  
	return num / denom

@numba.jit
def W_rot(l1, l2,  n,  dl):

	i1 = ltoi(l1, n, dl)
	i2 = ltoi(l2, n, dl)

	phi_l1l2 = anglebetween(l1, l2)

	output =  2. * np.cos(2.0 * phi_l1l2) 

	return output

@numba.jit
def W_rot2(l1, l2,  n,  dl):

	i1 = ltoi(l1, n, dl)
	i2 = ltoi(l2, n, dl)

	phi_l1l2 = anglebetween(l1, l2)

	output =  2. * np.sin(2.0 * phi_l1l2) 

	return output

# ===========================
# ==== INTEGRAL at L ========
# ===========================

@numba.jit#(nopython=1)
def ennellzero_EB_integral(eepowerspec2d, bbpowerspec2d, eetotallenspowerspec2d, bbtotallenspowerspec2d, dl, n, bigell):
	runningtotal = 0.
	usedvals = 0
 
	for i1x in range(n): 
		for i1y in range(n): 
			l1 = itol((i1x,i1y), n, dl)
			l2 = (bigell[0] - l1[0], bigell[1] - l1[1])
			i2 = ltoi(l2, n, dl)
		
			if not(circlecheck(l1, 3000, 0)): continue
			if not(circlecheck(l2, 3000, 0)): continue
				
			if (i1x < 0 or i1x >= n or i2[0] < 0 or i2[0] >= n or i1y < 0 or i1y >= n or i2[1] < 0 or i2[1] >= n ):
				continue

			smallfval = smallf_EB(l1, l2, eepowerspec2d, bbpowerspec2d, n, dl )
			filterval = filter_EB(l1, l2, eepowerspec2d, bbpowerspec2d, eetotallenspowerspec2d, bbtotallenspowerspec2d, n, dl )
	  
			runningtotal += smallfval * filterval
			usedvals += 1
  
	runningtotal *= dl * dl / (2*np.pi) / (2*np.pi)
	
	if runningtotal == 0.:
		return 0.
	else:
		return 1.0 / runningtotal

@numba.jit#(nopython=1)
def ennellzero_TB_integral(tepowerspec2d, tttotallenspowerspec2d, bbtotallenspowerspec2d, dl, n, bigell):
	runningtotal = 0.
	usedvals = 0
  
	for i1x in range(n): 
		for i1y in range(n): 
			l1 = itol((i1x,i1y), n, dl)
			l2 = (bigell[0] - l1[0], bigell[1] - l1[1])
			i2 = ltoi(l2, n, dl)
		
			if not(circlecheck(l1, 3000, 0)): continue
			if not(circlecheck(l2, 3000, 0)): continue
				
			if (i1x < 0 or i1x >= n or i2[0] < 0 or i2[0] >= n or i1y < 0 or i1y >= n or i2[1] < 0 or i2[1] >= n ):
				continue

			smallfval = smallf_TB(l1, l2, tepowerspec2d, n, dl )
			filterval = filter_TB(l1, l2, tepowerspec2d, tttotallenspowerspec2d, bbtotallenspowerspec2d, n, dl )
	  
			runningtotal += smallfval * filterval
			usedvals += 1
  
	runningtotal *= dl * dl / (2*np.pi) / (2*np.pi)
	
	if runningtotal == 0.:
		return 0.
	else:
		return 1.0 / runningtotal
	
@numba.jit
def Cl_BB_rot_integral(eepowerspec2d, aapowerspec2d, dl, n, l):
	runningtotal = 0.
	usedvals = 0

	il = ltoi(l, n, dl)

	for ipx in range(n): 
		for ipy in range(n): 
			lp = itol((ipx,ipy), n, dl)
			l_min_lp = (l[0] - lp[0], l[1] - lp[1])
			il_min_ilp = ltoi(l_min_lp, n, dl)
		
			if not(circlecheck(lp, 3000, 0)): continue
			if not(circlecheck(l_min_lp, 3000, 0)): continue
				
			if (ipx < 0 or ipx >= n or il_min_ilp[0] < 0 or il_min_ilp[0] >= n or ipy < 0 or ipy >= n or il_min_ilp[1] < 0 or il_min_ilp[1] >= n ):
				continue
			
			runningtotal += eepowerspec2d[ipx,ipy] * aapowerspec2d[il_min_ilp[0]][il_min_ilp[1]] * W_rot(l, lp,  n,  dl )**2
			usedvals += 1
  
	runningtotal *= dl * dl / (2*np.pi) / (2*np.pi)
	
	if runningtotal == 0.:
		return 0.
	else:
		return runningtotal

@numba.jit
def Cl_BB_rot_integral2(bbpowerspec2d, aapowerspec2d, dl, n, l):
	runningtotal = 0.
	usedvals = 0

	il = ltoi(l, n, dl)

	for ipx in range(n): 
		for ipy in range(n): 
			lp = itol((ipx,ipy), n, dl)
			l_min_lp = (l[0] - lp[0], l[1] - lp[1])
			il_min_ilp = ltoi(l_min_lp, n, dl)
		
			if not(circlecheck(lp, 3000, 0)): continue
			if not(circlecheck(l_min_lp, 3000, 0)): continue
				
			if (ipx < 0 or ipx >= n or il_min_ilp[0] < 0 or il_min_ilp[0] >= n or ipy < 0 or ipy >= n or il_min_ilp[1] < 0 or il_min_ilp[1] >= n ):
				continue
			
			runningtotal += bbpowerspec2d[ipx,ipy] * aapowerspec2d[il_min_ilp[0]][il_min_ilp[1]] * W_rot2(l, lp,  n,  dl )**2
			usedvals += 1
  
	runningtotal *= dl * dl / (2*np.pi) / (2*np.pi)
	
	if runningtotal == 0.:
		return 0.
	else:
		return runningtotal

@numba.jit#(nopython=1)
def Cl_BB_rot_res_integral(eepowerspec2d, aapowerspec2d, eetotallenspowerspec2d, aatotalpowerspec2d, dl, n, l):
	runningtotal = 0.
	usedvals = 0

	il = ltoi(l, n, dl)

	for ipx in range(n): 
		for ipy in range(n): 
			lp = itol((ipx,ipy), n, dl)
			l_min_lp = (l[0] - lp[0], l[1] - lp[1])
			il_min_ilp = ltoi(l_min_lp, n, dl)
		
			if not(circlecheck(lp, 3000, 0)): continue
			if not(circlecheck(l_min_lp, 3000, 0)): continue
				
			if (ipx < 0 or ipx >= n or il_min_ilp[0] < 0 or il_min_ilp[0] >= n or ipy < 0 or ipy >= n or il_min_ilp[1] < 0 or il_min_ilp[1] >= n ):
				continue
			
			runningtotal += eepowerspec2d[ipx,ipy] * aapowerspec2d[il_min_ilp[0]][il_min_ilp[1]] * W_rot(l, lp,  n,  dl )**2 * (1. - (eepowerspec2d[ipx][ipy]/eetotallenspowerspec2d[ipx][ipy])*(aapowerspec2d[il_min_ilp[0]][il_min_ilp[1]]/aatotalpowerspec2d[il_min_ilp[0]][il_min_ilp[1]]) )
			usedvals += 1
  
	runningtotal *= dl * dl / (2*np.pi) / (2*np.pi)
	
	if runningtotal == 0.:
		return 0.
	else:
		return runningtotal

# ===========================
# ======= INTEGRALS ========
# ===========================

@numba.jit
def ennellzero_EB(eepowerspec2d, bbpowerspec2d, eetotallenspowerspec2d, bbtotallenspowerspec2d, dl, n, nwanted,):
	output = np.zeros(nwanted)
  
	for ibigell in range(1,nwanted):
		bigell = (dl * ibigell, 0)
		output[ibigell] = ennellzero_EB_integral(eepowerspec2d, bbpowerspec2d, eetotallenspowerspec2d, bbtotallenspowerspec2d, dl, n, bigell)

	return output

@numba.jit
def ennellzero_TB(tepowerspec2d, tttotallenspowerspec2d, bbtotallenspowerspec2d, dl, n, nwanted):
	output = np.zeros(nwanted)

	for ibigell in range(1,nwanted):
		bigell = (dl * ibigell, 0)
		output[ibigell] = ennellzero_TB_integral(tepowerspec2d, tttotallenspowerspec2d, bbtotallenspowerspec2d, dl, n, bigell)

	return output

@numba.jit
def ClBB_rot(eepowerspec2d, aapowerspec2d, dl, n, nwanted,):
	output = np.zeros(nwanted)
  
	for iell in range(1,nwanted):
		ell = (dl * iell, 0)
		output[iell] = Cl_BB_rot_integral(eepowerspec2d, aapowerspec2d, dl, n, ell)

	return output

@numba.jit
def ClBB_rot2(bbpowerspec2d, aapowerspec2d, dl, n, nwanted,):
	output = np.zeros(nwanted)
  
	for iell in range(1,nwanted):
		ell = (dl * iell, 0)
		output[iell] = Cl_BB_rot_integral2(bbpowerspec2d, aapowerspec2d, dl, n, ell)

	return output

@numba.jit
def ClBB_rot_res(eepowerspec2d, aapowerspec2d, eetotallenspowerspec2d, aatotalpowerspec2d, dl, n, nwanted,):
	output = np.zeros(nwanted)
  
	for iell in range(1,nwanted):
		ell = (dl * iell, 0)
		output[iell] = Cl_BB_rot_res_integral(eepowerspec2d, aapowerspec2d, eetotallenspowerspec2d, aatotalpowerspec2d, dl, n, ell)

	return output


# ===========================
# ======= HELPERS ===========
# ===========================

def GimmeNl(cmbspec, beam, noise, est='EB', dl=8, n=512, nwanted=200, f_delens=0., lmax_delens=5000, lknee=None, alpha=None):
	lxgrid, lygrid  = np.meshgrid( np.arange(-n/2.,n/2.)*dl, np.arange(-n/2.,n/2.)*dl )
	lgrid = np.sqrt(lxgrid**2 + lygrid**2)
	L     = np.arange(0,nwanted)*dl    
	ell   = np.arange(0,2001)

	nltt = nl_cmb(noise           , beam, cmbspec.shape[0]-1, alpha=alpha, lknee=lknee)
	nlpp = nl_cmb(noise*np.sqrt(2), beam, cmbspec.shape[0]-1, alpha=alpha, lknee=lknee)
	
	if est == 'EB':
		clee = cmbspec[:,1].copy()
		clbb = cmbspec[:,2].copy()
		clbb[:lmax_delens+1] = (1.-f_delens)*clbb[:lmax_delens+1]
		
		eepowerspec2d          = np.interp(lgrid, np.arange(cmbspec.shape[0]), clee)
		bbpowerspec2d          = np.interp(lgrid, np.arange(cmbspec.shape[0]), clbb)
		nlpp2d                 = np.interp(lgrid, np.arange(cmbspec.shape[0]), nlpp) 
		eetotallenspowerspec2d = eepowerspec2d.copy() + nlpp2d.copy()
		bbtotallenspowerspec2d = bbpowerspec2d.copy() + nlpp2d.copy()

		nl = ennellzero_EB(eepowerspec2d, bbpowerspec2d, eetotallenspowerspec2d, bbtotallenspowerspec2d, dl, n, nwanted)
		# nl = interpolate.interp1d(L,ennellzero_EB(eepowerspec2d, bbpowerspec2d, eetotallenspowerspec2d, bbtotallenspowerspec2d, dl, n, nwanted), fill_value='extrapolate')
	elif est == 'TB': 
		cltt = cmbspec[:,0].copy()
		clte = cmbspec[:,3].copy()
		clbb = cmbspec[:,2].copy()
		clbb[:lmax_delens+1] = (1.-f_delens)*clbb[:lmax_delens+1]
		
		ttpowerspec2d          = np.interp(lgrid, np.arange(cmbspec.shape[0]), cltt)
		tepowerspec2d          = np.interp(lgrid, np.arange(cmbspec.shape[0]), clte)
		bbpowerspec2d          = np.interp(lgrid, np.arange(cmbspec.shape[0]), clbb)
		nltt2d                 = np.interp(lgrid, np.arange(cmbspec.shape[0]), nltt) 
		nlpp2d                 = np.interp(lgrid, np.arange(cmbspec.shape[0]), nlpp) 
		tttotallenspowerspec2d = ttpowerspec2d.copy() + nltt2d.copy()
		bbtotallenspowerspec2d = bbpowerspec2d.copy() + nlpp2d.copy()
		
		nl = ennellzero_TB(tepowerspec2d, tttotallenspowerspec2d, bbtotallenspowerspec2d, dl, n, nwanted)
	else:
		ValueError
	
	return L, nl

def GimmeNl_at_L(el, cmbspec, beam, noise, est='EB', dl=8, n=512, f_delens=0., lmax_delens=5000):
	lxgrid, lygrid  = np.meshgrid( np.arange(-n/2.,n/2.)*dl, np.arange(-n/2.,n/2.)*dl )
	lgrid = np.sqrt(lxgrid**2 + lygrid**2)
	L     = np.arange(0,nwanted)*dl    

	nltt = nl_cmb(noise           , beam, cmbspec.shape[0]-1)
	nlpp = nl_cmb(noise*np.sqrt(2), beam, cmbspec.shape[0]-1)
	
	print 'f = %.2f and lmax = %d' %(f_delens, lmax_delens)
	
	if est == 'EB':
		clee = cmbspec[:,1].copy()
		clbb = cmbspec[:,2].copy()
		clbb[:lmax_delens+1] = (1.-f_delens)*clbb[:lmax_delens+1]
		# plt.plot(clbb)
		eepowerspec2d          = np.interp(lgrid, np.arange(cmbspec.shape[0]), clee)
		bbpowerspec2d          = np.interp(lgrid, np.arange(cmbspec.shape[0]), clbb)
		nlpp2d                 = np.interp(lgrid, np.arange(cmbspec.shape[0]), nlpp) 
		eetotallenspowerspec2d = eepowerspec2d.copy() + nlpp2d.copy()
		bbtotallenspowerspec2d = bbpowerspec2d.copy() + nlpp2d.copy()

		nl = ennellzero_EB_integral(eepowerspec2d, bbpowerspec2d, eetotallenspowerspec2d, bbtotallenspowerspec2d, dl, n, (el,0))
	elif est == 'TB': 
		cltt = cmbspec[:,0].copy()
		clte = cmbspec[:,3].copy()
		clbb = cmbspec[:,2].copy()
		clbb[:lmax_delens+1] = (1.-f_delens)*clbb[:lmax_delens+1]
		
		ttpowerspec2d          = np.interp(lgrid, np.arange(cmbspec.shape[0]), cltt)
		tepowerspec2d          = np.interp(lgrid, np.arange(cmbspec.shape[0]), clte)
		bbpowerspec2d          = np.interp(lgrid, np.arange(cmbspec.shape[0]), clbb)
		nltt2d                 = np.interp(lgrid, np.arange(cmbspec.shape[0]), nltt) 
		nlpp2d                 = np.interp(lgrid, np.arange(cmbspec.shape[0]), nlpp) 
		tttotallenspowerspec2d = ttpowerspec2d.copy() + nltt2d.copy()
		bbtotallenspowerspec2d = bbpowerspec2d.copy() + nlpp2d.copy()
		
		nl = ennellzero_TB_integral(tepowerspec2d, tttotallenspowerspec2d, bbtotallenspowerspec2d, dl, n, (el,0))
	else:
		ValueError
	
	return nl

def GimmeClBBRot(cmbspec, kind='default', A_CB=1., nu=30., B=1., H_I=1., f_a=1., dl=8, n=512, nwanted=200):
	lxgrid, lygrid  = np.meshgrid( np.arange(-n/2.,n/2.)*dl, np.arange(-n/2.,n/2.)*dl )
	lgrid = np.sqrt(lxgrid**2 + lygrid**2)
	L     = np.arange(0,nwanted)*dl    
	
	clee = cmbspec[:,1].copy()
	
	eepowerspec2d = np.interp(lgrid, np.arange(cmbspec.shape[0]), clee)
	
	ell = np.arange(cmbspec.shape[0])
	
	if kind == 'default':
		claa = np.nan_to_num(A_CB*1e-5*2*np.pi/ell/(ell+1))
	elif (kind == 'pmf') or (kind == 'PMF'):
		claa = np.nan_to_num(2.3e-5 * (30./nu)**4 * B**2 * 2*np.pi/ell/(ell+1))
	elif (kind == 'pseudoscalar') or (kind == 'CS'):
		claa = np.nan_to_num((H_I/2/np.pi/f_a)**2 * 2*np.pi/ell/(ell+1))

	claa[0] = 0.				

	aapowerspec2d = np.interp(lgrid, np.arange(cmbspec.shape[0]), claa)
	
	clrot = ClBB_rot(eepowerspec2d, aapowerspec2d, dl, n, nwanted)
	
	return L, clrot

def GimmeClBBRot2(cmbspec, kind='default', A_CB=1., nu=30., B=1., H_I=1., f_a=1., dl=8, n=512, nwanted=200):
	lxgrid, lygrid  = np.meshgrid( np.arange(-n/2.,n/2.)*dl, np.arange(-n/2.,n/2.)*dl )
	lgrid = np.sqrt(lxgrid**2 + lygrid**2)
	L     = np.arange(0,nwanted)*dl    
	
	clbb = cmbspec[:,2].copy()
	
	bbpowerspec2d = np.interp(lgrid, np.arange(cmbspec.shape[0]), clbb)
	
	ell = np.arange(cmbspec.shape[0])
	
	if kind == 'default':
		claa = np.nan_to_num(A_CB*1e-5*2*np.pi/ell/(ell+1))
	elif (kind == 'pmf') or (kind == 'PMF'):
		claa = np.nan_to_num(2.3e-5 * (30./nu)**4 * B**2 * 2*np.pi/ell/(ell+1))
	elif (kind == 'pseudoscalar') or (kind == 'CS'):
		claa = np.nan_to_num((H_I/2/np.pi/f_a)**2 * 2*np.pi/ell/(ell+1))

	claa[0] = 0.				

	aapowerspec2d = np.interp(lgrid, np.arange(cmbspec.shape[0]), claa)
	
	clrot = ClBB_rot2(bbpowerspec2d, aapowerspec2d, dl, n, nwanted)
	
	return L, clrot

def GimmeClBBRes(cmbspec, beam, noise, lnlpp=None, nlpp=None, dl=8, n=512, nwanted=200, kind='default', 
				 A_CB=1., nu=30., B=1., H_I=1., f_a=1., f_delens=0., lmax_delens=5000, 
				 lknee=None, alpha=None):
	lxgrid, lygrid  = np.meshgrid( np.arange(-n/2.,n/2.)*dl, np.arange(-n/2.,n/2.)*dl )
	lgrid = np.sqrt(lxgrid**2 + lygrid**2)
	L     = np.arange(0,nwanted)*dl    

	nlee = nl_cmb(noise*np.sqrt(2), beam, cmbspec.shape[0]-1, lknee=lknee, alpha=alpha)
	clee = cmbspec[:,1].copy()
	
	eepowerspec2d          = np.interp(lgrid, np.arange(cmbspec.shape[0]), clee)
	nlee2d                 = np.interp(lgrid, np.arange(cmbspec.shape[0]), nlee) 
	eetotallenspowerspec2d = eepowerspec2d.copy() + nlee2d.copy()

	ell = np.arange(cmbspec.shape[0])
	if kind == 'default':
		claa = np.nan_to_num(A_CB*1e-5*2*np.pi/ell/(ell+1))
	elif (kind == 'pmf') or (kind == 'PMF'):
		claa = np.nan_to_num(2.3e-5 * (30./nu)**4 * B**2 * 2*np.pi/ell/(ell+1))
	elif (kind == 'pseudoscalar') or (kind == 'CS'):
		claa = np.nan_to_num((H_I/2/np.pi/f_a)**2 * 2*np.pi/ell/(ell+1))

	claa[0] = 0.				

	laa, nlaa = GimmeNl(cmbspec, beam, noise, est='EB', dl=dl, n=n, nwanted=nwanted, f_delens=f_delens, lmax_delens=lmax_delens, lknee=lknee, alpha=alpha)
	aapowerspec2d = np.interp(lgrid, np.arange(cmbspec.shape[0]), claa)
	aatotalpowerspec2d = aapowerspec2d.copy() + np.interp(lgrid, laa, nlaa)

	clres = ClBB_rot_res(eepowerspec2d, aapowerspec2d, eetotallenspowerspec2d, aatotalpowerspec2d, dl, n, nwanted)
	
	return L, clres

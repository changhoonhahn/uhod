'''

module for interfacing with CAMELS data 


'''
import os
import numpy as np 
import astropy.table as aTable 


def Rockstar(snap, real='LH_0', sim='tng') : 
    ''' rockstar+consistent tree halo catalogs with assembly history information 
    '''
    # rockstar directory 
    dat_dirs = ['/projects/QUIJOTE/CAMELS/Rockstar/']
    sims = {'tng': 'IllustrisTNG', 
            'tng_dm': 'IllustrisTNG_DM', 
            'simba': 'SIMBA', 
            'simba_dm': 'SIMBA_DM'} 

    for _dir in dat_dirs: 
        dat_dir = os.path.join(_dir, sims[sim], real, 'hlists') 
        if not os.path.isdir(_dir):  
            continue 
    
    asnap = a_snap(snap) 
    fsnap = 'hlist_%.5f.list' % asnap

    tab = aTable.Table.read(os.path.join(dat_dir, fsnap), format='ascii') 
    
    # clean up the column names 
    for col in tab.colnames: 
        if '(500c)' in col: 
            tab.rename_column(col, col.replace('(500c)', '_500c').split('(')[0])
        else: 
            tab.rename_column(col, col.split('(')[0])
    return tab


def AHF(snap, real='LH_0', sim='tng'): 
    ''' amiga halo finder given snapshot, realization, and simulation 
    '''
    # AHF directory
    ahf_dir = dat_dir(real=real, sim=sim)
    
    zsnap = z_snap(snap) 
    fsnap = 'snap%sRpep..0000.z%.3f.AHF_halos' % (str(snap).zfill(3), zsnap)
    
    tab = aTable.Table.read(os.path.join(ahf_dir, fsnap), format='ascii')

    for col in tab.colnames: 
        tab.rename_column(col, col.split('(')[0])
    return tab


def LHC(sim='tng'): 
    ''' retrun Omega_m sigma_8 A1 A2 A3 A4 values of latin hyper cube
    '''
    _DAT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dat')
    
    return np.loadtxt(os.path.join(_DAT_DIR, '%s_lhc_params.txt' % sim)) 


def dat_dir(real='LH_0', sim='tng'):
    ''' directory of CAMELS realization for given sim 
    '''
    dat_dirs = [
            '/projects/QUIJOTE/CAMELS/AHF/',
            '/Users/chahah/data/CAMELS/Sims/' # mbp
            ]
    sims = {'tng': 'IllustrisTNG', 
            'simba': 'SIMBA'} 

    for _dir in dat_dirs: 
        dat_dir = os.path.join(_dir, sims[sim], real) 
        if os.path.isdir(_dir): return dat_dir 


def z_snap(snap): 
    ''' redshift for given snapshot
    '''
    z_dict = {
            32: 0.049, 
            33: 0.
            }
    return z_dict[snap]


def a_snap(snap): 
    ''' scale factor for given snapshot
    '''
    a_dict = {
            32: 0.95332, 
            33: 1.00000
            }
    return a_dict[snap]

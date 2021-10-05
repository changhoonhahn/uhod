'''

module for interfacing with CAMELS data 


'''
import os
import astropy.table as aTable 


def AHF(snap, real='LH_0', sim='tng'): 
    ''' amiga halo finder given snapshot, realization, and simulation 
    '''
    # AHF directory
    ahf_dir = os.path.join(dat_dir(real=real, sim=sim), 'AHF') 
    
    zsnap = z_snap(snap) 
    fsnap = 'snap%sRpep..0000.z%.3f.AHF_halos' % (str(snap).zfill(3), zsnap)
    
    tab = aTable.Table.read(os.path.join(ahf_dir, fsnap), format='ascii')

    for col in tab.colnames: 
        tab.rename_column(col, col.split('(')[0])
    return tab


def dat_dir(real='LH_0', sim='tng'):
    ''' directory of CAMELS realization for given sim 
    '''
    dat_dirs = [
            '/Users/chahah/data/CAMELS/Sims/' # mbp
            ]
    sims = {'tng': 'IllustrisTNG'} 

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

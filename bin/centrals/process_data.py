'''


scripts for processing the CAMELS rockstar data


'''
import os, sys
import numpy as np


task    = sys.argv[1]
sim     = sys.argv[2]


def compile_halos(sim='tng'): 
    ''' compile halos from rockstar catalogs
    '''
    import astropy.table as atable
    if sim == 'tng': 
        frock = '/tigress/chhahn/CAMELS/tng.rockstar.dat' 
    else: 
        raise NotImplementedError

    rockstar = atable.Table.read(frock, format='ascii')

    rockstar['logMvir'] = np.log10(rockstar['Mvir'])
    rockstar['M_star'] = rockstar['SM']
    rockstar['logMstar'] = np.log10(rockstar['M_star'])
    rockstar['concentration'] = rockstar['Rvir'] / rockstar['rs']

    rockstar['logVmax'] = np.log10(rockstar['vmax'])
    rockstar['logVacc'] = np.log10(rockstar['Vacc'])
    rockstar['logVpeak'] = np.log10(rockstar['Vpeak'])

    rockstar['logMacc'] = np.log10(rockstar['Macc'])
    rockstar['logMpeak'] = np.log10(rockstar['Mpeak'])

    is_halo = (rockstar['pid'] == -1) # halo not a subhalo

    prop_internal = ['logMstar', 'logMvir', 'logVmax', 'Spin', 'concentration', 'b_to_a', 'c_to_a', 'Xoff', 'Voff', 'Rvir', 'rs', 'Vrms']
    prop_assembly = ['logMacc', 'logVacc', 'logMpeak', 'logVpeak', 'Acc_Rate_Inst', 'Acc_Rate_1*Tdyn', 'Tidal_Force_Tdyn']
    prop_theta = ['Om', 's8', 'Asn1', 'Asn2', 'Aagn1', 'Aagn2']
    
    # read in data 
    f = h5py.File(f'/tigress/chhahn/CAMELS/{sim}.rockstar.halos.hdf5', 'w')
    for col in prop_internal + prop_assembly + prop_theta: 
        f.create_dataset(col, data=np.array(rockstar[col][is_halo].data))
    f.close()
    return None


if task == 'compile_halos': 
    compile_halos(sim) 

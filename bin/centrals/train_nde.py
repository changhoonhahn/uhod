'''

script to train NDEs 


'''
import os, sys
import h5py 
import numpy as np 

import torch
from torch.utils.tensorboard.writer import SummaryWriter

from sbi import utils as Ut
from sbi import inference as Inference


def train_p_dv(sim, gpu=True): 
    ''' train p(delta v | theta_halo, theta_cosmo, theta_baryon) 
    '''
    if gpu: 
        device = torch.device('cuda') 
    else: 
        device = torch.device('cpu') 

    # read in data 
    f = h5py.File(f'/tigress/chhahn/CAMELS/{sim}.rockstar.halos.hdf5', 'r')
    
    # restrict to halos above 10^10 Msun and with central galaxies
    cut = np.array(halos['logMvir'][...] > 10.) & (halos['logMstar'][...] > 0.)
    
    y_train = halos['Voff'][...][cut]
    x_train = np.concatenate([
        halos[col][...][cut,None] for col in 
        ['logMvir', 'Spin', 'concentration', 'Om', 's8', 'Asn1', 'Asn2', 'Aagn1', 'Aagn2']])

    # set prior
    lower_bounds = torch.tensor([0])
    upper_bounds = torch.tensor([1e3])
    prior = Ut.BoxUniform(low=lower_bounds, high=upper_bounds, device=device)

    # train NDEs 
    anpes, phats, archs = [], [], []
    valid_logprobs, best_valid_logprobs = [], []
    for i in range(n_model):
        nhidden = int(np.ceil(np.exp(np.random.uniform(np.log(64), np.log(512)))))
        nblocks = int(np.random.uniform(3, 10))
        print('MAF with nhidden=%i; nblocks=%i' % (nhidden, nblocks))
        archs.append('%ix%i' % (nhidden, nblocks))

        anpe = Inference.SNPE(
                prior=prior,
                density_estimator=Ut.posterior_nn('maf', hidden_features=nhidden, num_transforms=nblocks),
                device=device,
                summary_writer=SummaryWriter(
                    '/tigress/chhahn/CAMELS/centrals/nde.%s.p_dv.%ix%i.%i' % (sim, nhidden, nblocks, i)))
        anpe.append_simulations(
            torch.as_tensor(y_train.astype(np.float32)),
            torch.as_tensor(x_train.astype(np.float32)))

        phat = anpe.train()

        nde = anpe.build_posterior(phat)

        fanpe = '/tigress/chhahn/CAMELS/centrals/nde.%s.p_dv.%ix%i.%i.pt' % (sim, nhidden, nblocks, i)
        torch.save(nde, fanpe)
        np.save(fanpe.replace('.pt', '.valid_loss.npy'), np.array(anpe._summary['validation_log_probs']))

        anpes.append(anpe)
        phats.append(phat)

        valid_logprobs.append(anpe._summary['validation_log_probs'])
        best_valid_logprobs.append(anpe._summary['best_validation_log_probs'])

    ibest = np.argmax(best_valid_logprobs)
    best_anpe = anpes[ibest]
    best_phat = phats[ibest]
    best_arch = archs[ibest]

    best_nde = best_anpe.build_posterior(best_phat)

    #save trained ANPE
    fanpe = '/tigress/chhahn/CAMELS/centrals/nde.%s.p_dv.best.pt' % sim 
    print('saving to %s' % fanpe)
    torch.save(best_nde, fanpe)
    np.save(fanpe.replace('.pt', '.valid_loss.npy'), np.array(valid_logprobs[ibest]))
    return None 


def train_p_dx(sim, gpu=True): 
    ''' train p(delta v | theta_halo, theta_cosmo, theta_baryon) 
    '''
    if gpu: 
        device = torch.device('cuda') 
    else: 
        device = torch.device('cpu') 

    # read in data 
    f = h5py.File(f'/tigress/chhahn/CAMELS/{sim}.rockstar.halos.hdf5', 'r')
    
    # restrict to halos above 10^10 Msun and with central galaxies
    cut = np.array(halos['logMvir'][...] > 10.) & (halos['logMstar'][...] > 0.)
    
    y_train = halos['Voff'][...][cut]
    x_train = np.concatenate([
        halos[col][...][cut,None] for col in 
        ['logMvir', 'Spin', 'concentration', 'Om', 's8', 'Asn1', 'Asn2', 'Aagn1', 'Aagn2']])

    # set prior
    lower_bounds = torch.tensor([0])
    upper_bounds = torch.tensor([1e3])
    prior = Ut.BoxUniform(low=lower_bounds, high=upper_bounds, device=device)

    # train NDEs 
    anpes, phats, archs = [], [], []
    valid_logprobs, best_valid_logprobs = [], []
    for i in range(n_model):
        nhidden = int(np.ceil(np.exp(np.random.uniform(np.log(64), np.log(512)))))
        nblocks = int(np.random.uniform(3, 10))
        print('MAF with nhidden=%i; nblocks=%i' % (nhidden, nblocks))
        archs.append('%ix%i' % (nhidden, nblocks))

        anpe = Inference.SNPE(
                prior=prior,
                density_estimator=Ut.posterior_nn('maf', hidden_features=nhidden, num_transforms=nblocks),
                device=device,
                summary_writer=SummaryWriter(
                    '/tigress/chhahn/CAMELS/centrals/nde.%s.p_dv.%ix%i.%i' % (sim, nhidden, nblocks, i)))
        anpe.append_simulations(
            torch.as_tensor(y_train.astype(np.float32)),
            torch.as_tensor(x_train.astype(np.float32)))

        phat = anpe.train()

        nde = anpe.build_posterior(phat)

        fanpe = '/tigress/chhahn/CAMELS/centrals/nde.%s.p_dv.%ix%i.%i.pt' % (sim, nhidden, nblocks, i)
        torch.save(nde, fanpe)
        np.save(fanpe.replace('.pt', '.valid_loss.npy'), np.array(anpe._summary['validation_log_probs']))

        anpes.append(anpe)
        phats.append(phat)

        valid_logprobs.append(anpe._summary['validation_log_probs'])
        best_valid_logprobs.append(anpe._summary['best_validation_log_probs'])

    ibest = np.argmax(best_valid_logprobs)
    best_anpe = anpes[ibest]
    best_phat = phats[ibest]
    best_arch = archs[ibest]

    best_nde = best_anpe.build_posterior(best_phat)

    #save trained ANPE
    fanpe = '/tigress/chhahn/CAMELS/centrals/nde.%s.p_dv.best.pt' % sim 
    print('saving to %s' % fanpe)
    torch.save(best_nde, fanpe)
    np.save(fanpe.replace('.pt', '.valid_loss.npy'), np.array(valid_logprobs[ibest]))
    return None 


task    = sys.argv[1]
sim     = sys.argv[2]

if task == 'train_p_dv': 
    train_p_dv(gpu=True)

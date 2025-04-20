import pickle
import numpy as np
import math
from tqdm.auto import tqdm
import torch
import os
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from factor_analyzer import Rotator

sigmoid = nn.Sigmoid()
np_sigmoid = lambda x: 1/(1+np.exp(-np.clip(x,-30,30)))
np_logit = lambda x: np.log(x/(1-x))

def create_irt_train_mask(Y, validation_fraction, random_seed):
    mask = torch.ones(Y.shape, dtype=torch.bool).reshape(-1)
    local_state = np.random.RandomState(random_seed)
    mask[local_state.choice(len(mask), int(validation_fraction*len(mask)+1))] = False
    mask = mask.reshape(Y.shape)
    return mask

def loss_matrix(Y, P, beta=.05):
    return (Y-P).abs() #F.huber_loss(P, Y, reduction='none', delta=beta)

def np_irt_forward(Theta, Alpha, beta, kappa):
    return kappa.T + (1-kappa.T)*np_sigmoid(Theta@Alpha.T-beta.T)
    
def irt_forward(Theta, Alpha, beta, kappa):
    return kappa.T + (1-kappa.T)*sigmoid(Theta@Alpha.T-beta.T)
    

def fit_IRT(Y,
            d,
            mask,
            Alpha=None,
            beta=None,
            Theta=None,
            kappa=None,
            lr = 1,
            n_epochs = 10000,
            validation_fraction= .1,
            tol = 1e-5,
            model='m3pl',
            max_asymp = .5,
            early_stop_patience = 50,
            scheduler_factor = 0.95,
            scheduler_patience = 10,
            val_step = 2,
            print_every = 100,
            random_seed = 42,
            verbose = False,
            device='cpu'):

    assert val_step<=print_every
    assert model in ['m2pl','m3pl']

    if model!='m3pl': max_asymp = 0
        
    ### Basic defs
    C = 5*(d**.5)
    mask = torch.tensor(mask, requires_grad=False)
    if validation_fraction>0: 
        train_mask = mask*create_irt_train_mask(Y, validation_fraction, random_seed)
        val_mask = mask*(~train_mask)
    Y = torch.tensor(Y, requires_grad=False).float().to(device)
    Y[~mask] = -999
    n_llms = Y.shape[0]

    ### Defining training variables
    parameters = []
    torch.manual_seed(random_seed)
    
    #beta
    if beta is None:
        beta = torch.nn.Parameter(torch.normal(0, .1, size=(Y.shape[1],1,), dtype=torch.float32, device=device))
        parameters.append(beta)
    else:
        beta = torch.tensor(beta, requires_grad=False).float().to(device)

    #kappa
    if kappa is None:
        kappa = torch.nn.Parameter(torch.full((Y.shape[1], 1), .2, dtype=torch.float32, device=device))
        parameters.append(kappa)
    else:
        kappa = torch.tensor(kappa, requires_grad=False).float().to(device)
        
    #Alpha
    if Alpha is None:
        scale = 1/(d**.5)
        Alpha = torch.nn.Parameter(torch.normal(0, scale, size=(Y.shape[1],d,), dtype=torch.float32, device=device)) 
        parameters.append(Alpha)
    else:
        Alpha = torch.tensor(Alpha, requires_grad=False).float().to(device)

    #Theta
    if Theta is None:
        Theta = torch.nn.Parameter(torch.normal(0, 1/(d**.5), size=(n_llms,d,), dtype=torch.float32, device=device))
        parameters.append(Theta)
    else:
        Theta = torch.tensor(Theta, requires_grad=False).float().to(device)
    
    ### Training
    optimizer = torch.optim.Adam(parameters, lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience)
    
    # Early stopping parameters
    best_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    early_stop = False
    
    # Losses
    train_losses =[]
    val_losses =[]
    val_accs =[]
    val_maes = []

    ### Defining sub-function for optimization
    for epoch in tqdm(range(n_epochs), desc=f"Training IRT with d={d} and lr={lr}.", disable=not verbose):
        optimizer.zero_grad()
        P = irt_forward(Theta, Alpha, beta, kappa)
        if validation_fraction>0: loss = loss_matrix(Y, P)[train_mask].mean()
        else: loss = loss_matrix(Y, P)[mask].mean()
        loss.backward()
        optimizer.step()

        # Projection step
        with torch.no_grad():
            # Items
            norms = torch.norm(torch.hstack((Alpha,beta)), dim=1)
            idx_proj = norms>C
            Alpha[idx_proj] = C*Alpha[idx_proj]/(norms[idx_proj][:,None])
            beta[idx_proj] = C*beta[idx_proj]/(norms[idx_proj][:,None])
            kappa.clamp_(0, max_asymp)
            
            # Test takes
            C2 = (C**2-1)**.5
            norms = torch.norm(Theta, dim=1)
            idx_proj = norms>C2
            Theta[idx_proj] = C2*Theta[idx_proj]/(norms[idx_proj][:,None])
                            
        train_losses.append(loss.item())

        if epoch%val_step==0:
            with torch.no_grad():
                if validation_fraction>0:
                    P = irt_forward(Theta, Alpha, beta, kappa)
                    loss = loss_matrix(Y, P)[val_mask].mean()
                    val_losses.append(loss.item())
                    val_accs.append(((Y>.5)==(P>.5))[val_mask].float().mean().item())
                    val_maes.append((Y-P)[val_mask].float().abs().mean().item())
                    scheduler.step(val_losses[-1])

                    # Check for early stopping
                    if val_losses[-1] + tol < best_loss:
                        best_loss = val_losses[-1]
                        val_mae = val_maes[-1]
                        best_epoch = epoch
                        epochs_no_improve = 0

                        best_beta = beta.detach().cpu().numpy()
                        best_Alpha = Alpha.detach().cpu().numpy()
                        best_Theta = Theta.detach().cpu().numpy()
                        best_kappa = kappa.detach().cpu().numpy()
                    else:
                        epochs_no_improve += 1
        
                    if epochs_no_improve >= early_stop_patience:
                        if verbose: print(f"Early stop at epoch {epoch} - best val loss {best_loss:.5f} - val MAE {val_mae:.5f} - best epoch {best_epoch}")
                        early_stop = True
                        break

                    if verbose:
                        if epoch%print_every==0:
                            tqdm.write(f"epoch={epoch:04d}, d={d}, tol={tol}, train loss={train_losses[-1]:.5f}, val loss={val_losses[-1]:.5f}, val MAE={val_maes[-1]:.5f}, val acc={val_accs[-1]:.5f}, lr={scheduler.optimizer.param_groups[0]['lr']:.5f}")
                
                else:
                     # Check for early stopping
                    scheduler.step(train_losses[-1])
                    if train_losses[-1] + tol < best_loss:
                        best_loss = train_losses[-1]
                        best_epoch = epoch
                        epochs_no_improve = 0

                        best_beta = beta.detach().cpu().numpy()
                        best_Alpha = Alpha.detach().cpu().numpy()
                        best_Theta = Theta.detach().cpu().numpy()
                        best_kappa = kappa.detach().cpu().numpy()
                    else:
                        epochs_no_improve += 1
        
                    if epochs_no_improve >= early_stop_patience:
                        if verbose: print(f"Early stop at epoch {epoch} - best train loss {best_loss:.5f} - best epoch {best_epoch}")
                        early_stop = True

                        break

                    if verbose:
                        if epoch%print_every==0:
                            tqdm.write(f"epoch={epoch:04d}, d={d}, tol={tol}, train loss={train_losses[-1]:.5f}, lr={scheduler.optimizer.param_groups[0]['lr']:.5f}")

            
    ### Output
    beta = best_beta
    Alpha = best_Alpha
    Theta = best_Theta
    kappa = best_kappa
    
    if validation_fraction>0:
        val_loss = best_loss
    else:
        val_loss = None

    if model=='m3pl': return Alpha, beta, Theta, kappa, val_loss
    else: return Alpha, beta, Theta, None, val_loss
    
class IRT:
    def __init__(self, ds=[1,2,3,5,10,15], model = 'm3pl', device='cpu'):

        self.ds = ds
        self.device = device
        self.model = model
 
    def fit(self,
            Y,
            mask,
            lrs=[1,.1,.01],
            n_epochs=10000,
            validation_fraction=.05,
            tol=1e-8,
            random_seed=42,
            verbose=True):

        assert validation_fraction>0 and validation_fraction<1

        self.best_loss = math.inf
        self.val_losses = {}
        
        for d in tqdm(self.ds, desc=f"Training IRT models for different d's."): 
            self.val_losses[d] = {}

            for lr in tqdm(lrs, desc=f"Training IRT models with d={d} and different lr's."):
                Alpha, beta, Theta, kappa, val_loss = fit_IRT(Y,
                                                              d,
                                                              mask,
                                                              Alpha=None,
                                                              beta=None,
                                                              Theta=None,
                                                              kappa=None,
                                                              model=self.model,
                                                              lr=lr,
                                                              n_epochs=n_epochs,
                                                              validation_fraction=validation_fraction,
                                                              tol=tol,
                                                              random_seed=random_seed,
                                                              verbose=verbose,
                                                              device=self.device)

                self.val_losses[d][lr] = val_loss
                
                if val_loss + tol < self.best_loss:
                    self.lr = lr
                    self.d = d
                    self.beta = beta
                    self.Alpha = Alpha
                    self.Theta = Theta
                    if self.model=='m3pl':
                        self.kappa = kappa
                    else:
                        self.kappa = None
                    self.best_loss = val_loss  

        self.stand(verbose=verbose)
        
        #if verbose: 
        print(f"\nBest d={self.d} - best val loss={self.best_loss:.5f}")
        
    def rotate(self):
        self.Theta_rot = {}
        self.Alpha_rot = {}
        self.beta_rot = {}
        for rot in ['varimax', 'oblimax', 'quartimax', 'equamax', 'geomin_ort', 'oblimin', 'quartimin', 'geomin_obl']:
            
            rotator = Rotator(method=rot).fit(self.Alpha)
            U = np.linalg.inv(rotator.rotation_).T
            self.Theta_rot[rot] = self.Theta@np.linalg.inv(U.T)
            self.Alpha_rot[rot] = (U.T@self.Alpha.T).T
            self.beta_rot[rot] = self.beta
            
            mu = self.Theta_rot[rot].mean(0)[None,:]
            sigma = self.Theta_rot[rot].std(0)[None,:]
            self.beta_rot[rot] -= self.Alpha_rot[rot]@mu.T
            self.Alpha_rot[rot] *= sigma
            self.Theta_rot[rot] = (self.Theta_rot[rot]-mu)/sigma  
            
    def stand(self, verbose=False):
        #standard
        if verbose:
            print(f"\nStandardizing Theta...")

        mu = self.Theta.mean(0)[None,:]
        cov = np.cov(self.Theta.T)
        sqrt_cov_inv = np.linalg.inv(np.linalg.cholesky(cov).T)
        Theta = ((self.Theta-mu)@sqrt_cov_inv)
        Alpha = (np.linalg.inv(sqrt_cov_inv)@self.Alpha.T).T
        beta = (self.beta.T-(mu@sqrt_cov_inv@Alpha.T)).T

        self.beta = beta
        self.Alpha = Alpha
        self.Theta = Theta
        
    def get_params(self):
        return self.Alpha, self.beta, self.kappa, self.Theta

    def fit_theta(self, Y, selected_items, lr=1, n_epochs=10000, tol=1e-5, random_seed = 42, verbose=True):
        
        _, _, Theta_test, _, _ = fit_IRT(Y,
                                         self.d,
                                         Alpha=self.Alpha[selected_items],
                                         beta=self.beta[selected_items],
                                         Theta=None,
                                         kappa=self.kappa[selected_items],
                                         model=self.model,
                                         lr=lr,
                                         n_epochs=n_epochs,
                                         validation_fraction=0,
                                         tol=tol,
                                         random_seed=random_seed,
                                         device=self.device,
                                         verbose=verbose)
        
        return {'new_Theta': Theta_test}

    def fit_alpha_beta_kappa(self, Y, selected_test_takers, lr=1, n_epochs=10000, tol=1e-5, random_seed = 42, verbose=True):
        
        Alpha, beta, _, kappa, _ = fit_IRT(Y,
                                           self.d,
                                           Alpha=None,
                                           beta=None,
                                           Theta=self.Theta[selected_test_takers],
                                           kappa=None,
                                           model=self.model,
                                           lr=lr,
                                           n_epochs=n_epochs,
                                           validation_fraction=0,
                                           tol=tol,
                                           random_seed=random_seed,
                                           device=self.device,
                                           verbose=verbose)

        return {'new_Alpha': Alpha, 'new_beta': beta, 'new_kappa': kappa}

    def save(self, path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(path, 'wb') as f:
            pickle.dump({
                'ds': self.ds,
                'device': self.device,
                'd': self.d,
                'beta': self.beta,
                'Alpha': self.Alpha,
                'kappa': self.kappa,
                'Theta': self.Theta,
                'best_loss': self.best_loss
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            params = pickle.load(f)
            self.ds = params['ds']
            self.device = params['device']
            self.d = params['d']
            self.beta = params['beta']
            self.Alpha = params['Alpha']
            self.kappa = params['kappa']
            self.Theta = params['Theta']
            self.best_loss = params['best_loss']
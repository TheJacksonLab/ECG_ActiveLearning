import torch
import numpy as np
import scipy.special as sc


def unc_sampler(predictions, gpnoise, globind, GP='EXACT'):
    """
    GP uncertainity query  Q_${unc}$(u) = argmin{xi elem u} \frac{|mu(xi)|}{ sqrt( var(xi) ) }
    Input Args:
    prediction : (tensor)
    gpnoise    : Learned GP noise variance (1x1 tensor)
    globind :(list or tuple) List of global dataset index for prediction
    GP : EXACT, MULTITASK (str)
    Return:
    index : Argmin  global index to be added to training
    """

    from itertools import compress

    unc = predictions.mean.abs() / torch.sqrt(predictions.variance + gpnoise)
    # gpnoise  ---> torch.exp(2 * gpnoise)
    # lowerbound = torch.abs( torch.mean(unc.norm(dim =1)) - torch.std(unc.norm(dim =1) ) )
    # trutharray = unc.norm(dim =1) < lowerbound
    # true_index = list(compress(range(len(trutharray)), trutharray))

    if GP == 'EXACT':
        selected_label = torch.argmin(unc)
    elif GP == 'MULTITASK':
        selected_label = torch.argmin(unc.norm(dim=1))

    index = []

    index.append(globind[selected_label])

    return index


class EMOC:
    """
    Expected Model Output Changes
    Freytag et al., 'Selecting Influential Examples: Active Learning with Expected Model Output Changes', ECCV (2014)
    """

    def __init__(self, predictions, model, gpnoise, globind, Ntrain, Nval, norm=1):
        """
        Input Args:
        prediction : (tensor)
        model : GPyTorch model after validation
        gpnoise    : Learned GP noise variance (1x1 tensor)
        globind :(list or tuple) List of global dataset index for prediction
        Ntrain : Number of training samples in current AL iteration (int)
        Nval : Number of validation samples in current AL iteration (int)
        norm : order of the moment (int)
        """
        self.predictions = predictions
        self.model = model
        self.Kernel = self.model.K
        self.gpnoise = gpnoise
        self.globind = globind
        self.Ntrain = Ntrain
        self.Nval = Nval
        self.norm = norm
        self.dtype = self.determine_dtype()
        self.emoc = torch.zeros(self.Nval).type(self.dtype)
        self.index = []

    def determine_dtype(self):
         """
         Determines the torch data type
         return:
         dtype : Tensor Float type
         """
         use_cuda = torch.cuda.is_available()
         dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
         
         return dtype

    def gaussianabsolutemoment(self):
        """
        Computes Gaussian absolute moments
        Return:
        moments : tensor
        """
        mean_to_variance = (self.predictions.mean ** 2) / self.predictions.variance 
        dim = mean_to_variance.shape[0]

        prefactor = ( (2 * self.predictions.variance  )**(self.norm/2.) * torch.lgamma(torch.tensor(1. + self.norm/2.)).exp() ) / (torch.sqrt(torch.tensor(np.pi) ) )

        hypf = sc.hyp1f1(np.ones(dim) * (- self.norm/2.), np.ones(dim) * 0.5, mean_to_variance * -0.5)

        return prefactor * hypf


    def compute_emoc(self):
        """
        Gather all the prefactor, test sample, kernels, and compute EMOC.
        Guide : delta alpha expanded as first term \times denominator  + second term \times denominator
        Return:
        index : Argmax  global index to be added to training
        """

        moments = self.gaussianabsolutemoment()

        K_inv_prod=torch.ones([self.Ntrain+1, self.Nval])*-1

        K_inv_prod[:self.Ntrain, :], _= torch.solve(torch.from_numpy(self.Kernel[0:self.Ntrain, self.Ntrain:].numpy(
        )), torch.from_numpy(self.Kernel[0:self.Ntrain, 0:self.Ntrain].numpy()) + torch.eye(self.Ntrain) * self.gpnoise)

        firstterm=torch.matmul(
            K_inv_prod[:-1, :].T, torch.from_numpy(self.Kernel[:self.Ntrain, :].numpy()))

        denominator=1 / (self.predictions.variance + self.gpnoise)
        
        # denominator[ind] *
        for ind in range(self.Nval):
            deltaF=denominator[ind] * firstterm[ind] + denominator[ind] * K_inv_prod[-1,
                ind] * torch.from_numpy(self.Kernel[self.Ntrain+ind, :].detach().numpy())
            self.emoc[ind] = torch.mean(torch.pow(torch.abs(deltaF), self.norm))

        self.emoc *= moments
        selected_label=torch.argmax(self.emoc)

        self.index.append(self.globind[selected_label])
        
        return self.index


class EMOC_GPU:
    """
    Expected Model Output Changes
    Freytag et al., 'Selecting Influential Examples: Active Learning with Expected Model Output Changes', ECCV (2014)
    """

    def __init__(self, predictions, model, gpnoise, globind, Ntrain, Nval, norm=1):
        """
        Input Args:
        prediction : (tensor)
        model : GPyTorch model after validation
        gpnoise    : Learned GP noise variance (1x1 tensor)
        globind :(list or tuple) List of global dataset index for prediction
        Ntrain : Number of training samples in current AL iteration (int)
        Nval : Number of validation samples in current AL iteration (int)
        norm : order of the moment (int)
        """
        self.predictions = predictions
        self.model = model
        self.Kernel = self.model.K
        self.gpnoise = gpnoise
        self.globind = globind
        self.Ntrain = Ntrain
        self.Nval = Nval
        self.norm = norm
        self.dtype = self.determine_dtype()
        self.emoc = torch.zeros(self.Nval).type(self.dtype)
        self.index = []

    def determine_dtype(self):
         """
         Determines the torch data type
         return:
         dtype : Tensor Float type
         """
         use_cuda = torch.cuda.is_available()
         dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
         
         return dtype

    def gaussianabsolutemoment(self):
        """
        Computes Gaussian absolute moments
        Return:
        moments : tensor
        """
        mean_to_variance = (self.predictions.mean ** 2) / self.predictions.variance 
        dim = mean_to_variance.shape[0]

        prefactor = ( (2 * self.predictions.variance )**(self.norm/2.) * torch.lgamma(torch.tensor(1. + self.norm/2.)).exp() ) / (torch.sqrt(torch.tensor(np.pi) ) )

        hypf = sc.hyp1f1(np.ones(dim) * (- self.norm/2.), np.ones(dim) * 0.5, mean_to_variance.cpu() * -0.5)
        hypf = hypf.cuda()
        return prefactor * hypf


    def compute_emoc(self):
        """
        Gather all the prefactor, test sample, kernels, and compute EMOC.
        Guide : delta alpha expanded as first term \times denominator  + second term \times denominator
        Return:
        index : Argmax  global index to be added to training
        """

        moments = self.gaussianabsolutemoment()

        K_inv_prod=torch.ones([self.Ntrain+1, self.Nval])*-1
        K_inv_prod = K_inv_prod.cuda()

        K_inv_prod[:self.Ntrain, :], _= torch.solve( torch.from_numpy(self.Kernel[0:self.Ntrain, self.Ntrain:].cuda().numpy(
        )).cuda() , torch.from_numpy(self.Kernel[0:self.Ntrain, 0:self.Ntrain].cuda().numpy()).cuda() + torch.eye(self.Ntrain).cuda() * self.gpnoise)

        firstterm=torch.matmul(
            K_inv_prod[:-1, :].T, torch.from_numpy(self.Kernel[:self.Ntrain, :].cuda().numpy()).cuda())

        denominator=1 / (self.predictions.variance + self.gpnoise)
        denominator = denominator.cuda()        
        # denominator[ind] *
        for ind in range(self.Nval):
            deltaF= denominator[ind] * firstterm[ind] + denominator[ind] * K_inv_prod[-1,
                ind] * torch.from_numpy(self.Kernel[self.Ntrain+ind, :].detach().cpu().numpy()).cuda()
            self.emoc[ind] = torch.mean(torch.pow(torch.abs(deltaF), self.norm))

        self.emoc *= moments
        selected_label=torch.argmax(self.emoc)

        self.index.append(self.globind[selected_label])
        
        return self.index


import torch 
import gpytorch
from ECGMOD.GPmodels import InferGPModel, estimate_gpnoise 

##DKL feature extractor
class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim, l1out, l2out, l3out, final):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, l1out)) #200
        self.add_module('batchnorm1', torch.nn.BatchNorm1d(l1out))   #200
        self.add_module('elu1', torch.nn.ELU())
        self.add_module('linear2', torch.nn.Linear(l1out, l2out) )  #200 , 100
        self.add_module('batchnorm2', torch.nn.BatchNorm1d(l2out)) #100 
        self.add_module('elu2', torch.nn.ELU())
        self.add_module('linear3', torch.nn.Linear(l2out, l3out)) #100, 50
        self.add_module('batchnorm3', torch.nn.BatchNorm1d(l3out)) #50 
        self.add_module('elu3', torch.nn.ELU())
        self.add_module('linear4', torch.nn.Linear(l3out, final)) #50 , 10 



class DKLRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel, data_dim, l1out, l2out, l3out, final):
        super(DKLRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if kernel == 'RBF':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=final))
        elif kernel == 'MATERN':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5,ard_num_dims=final))
        self.feature_extractor = LargeFeatureExtractor( data_dim, l1out, l2out, l3out, final)

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        # We're also scaling the features so that they're nice values
        projected_x = self.feature_extractor(x)
        projected_x = projected_x - projected_x.min(0)[0]
        projected_x = 2 * (projected_x / projected_x.max(0)[0]) - 1

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        self.K = covar_x
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



def DKLmodel(x_train, x_test, y_train, y_test, numy=1, n_iter=500,\
     verbose=True, GP='DKL',kernel='MATERN',lr=0.01,\
     l1out=1000, l2out=500, l3out=50, final=10):
    """
    A BO wrapper for  DKL  model
    Input Args:
    x_train : Train array (tensor)
    x_test : Test array (tensor)
    y_train : Train label (tensor)
    y_test : Test array (tensor)
    numy : number of y outputs (default = 1, int )
    n_iter : Number of iterations (default = 500, int)
    verbose : default = True (bool)
    GP : EXACT, MULTITASK, DKL (str)
    kernel : 'MATERN' (default) , RBF
    lr : learning rate (float)
    l1out, l2out, l3out, final : deep learning layer dimensions
    Return:
    mse, r2, gpnoise,  predictions, likelihood, model : float, float, tensor, tensor, _, _
    """

    if GP == "DKL" :
        likelihood = gpytorch.likelihoods.GaussianLikelihood( ) # noise_prior=gpytorch.priors.SmoothedBoxPrior(0.15, 1.5, sigma=0.5) 
        model = DKLRegressionModel(x_train, y_train, likelihood, kernel, x_train.size(-1), l1out, l2out, l3out, final)

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()


    #Define the parameters to optimize
    if GP == "DKL":
        optimizer = torch.optim.Adam([
    {'params': model.feature_extractor.parameters()},
    {'params': model.covar_module.parameters()},
    {'params': model.mean_module.parameters()},
    {'params': model.likelihood.parameters()},
], lr=lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)    

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    #training_iter = 500
    
    model.train()
    likelihood.train()

    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(x_train)
        loss = -mll(output, y_train)
        loss.backward()
        if verbose:
            if i % 100  == 0 or i == n_iter - 1:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, n_iter, loss.item()))
        optimizer.step()
        scheduler.step()

    mse, r2, predictions = InferGPModel(likelihood, model, x_test, y_test)
    gpnoise = estimate_gpnoise(likelihood, verbose=True)

    return mse, r2, gpnoise ,predictions, likelihood, model

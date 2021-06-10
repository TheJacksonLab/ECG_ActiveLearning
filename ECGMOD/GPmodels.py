"""
@author: gsivaraman@anl.gov
"""
import gpytorch
import torch 

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,numy,kernel):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=numy)
        if kernel == 'RBF':
            self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=numy, rank=1)
        elif kernel == 'MATERN':
            self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.MaternKernel(nu=2.5), num_tasks=numy, rank=1)
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        self.K = covar_x
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x,covar_x)
    
    

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel, data_dim):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if kernel == 'RBF':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=data_dim))  #45
        elif kernel == 'MATERN':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5,ard_num_dims=data_dim))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        self.K = covar_x
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


##DKL feature extractor 
class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, 1000))
        self.add_module('batchnorm1', torch.nn.BatchNorm1d(1000))
        self.add_module('elu1', torch.nn.ELU())
        self.add_module('linear2', torch.nn.Linear(1000, 500) )
        self.add_module('batchnorm2', torch.nn.BatchNorm1d(500))
        self.add_module('elu2', torch.nn.ELU())
        self.add_module('linear3', torch.nn.Linear(500, 100))    
        self.add_module('batchnorm3', torch.nn.BatchNorm1d(100))
        self.add_module('elu3', torch.nn.ELU())
        self.add_module('linear4', torch.nn.Linear(100, 10))



class DKLRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel, data_dim):
        super(DKLRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if kernel == 'RBF':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=10))
        elif kernel == 'MATERN':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5,ard_num_dims=10))
        self.feature_extractor = LargeFeatureExtractor( data_dim)

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



def GPmodel(x_train,x_test, y_train,y_test,numy=1,n_iter=500, verbose=True, GP='EXACT',kernel='MATERN'):
    """
    A method to manage various  GP model
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
    Return:
    mse, r2, gpnoise,  predictions, likelihood, model : float, float, tensor, tensor, _, _
    """

    if GP == 'MULTITASK':
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=numy)
        model = MultitaskGPModel(x_train, y_train, likelihood, numy, kernel)
    elif GP == 'EXACT' and numy == 1 : 
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(x_train, y_train, likelihood, kernel,  x_train.size(-1))
    elif GP == 'EXACT' and numy > 1 :     
        raise Exception("**ERROR** Multi Task GP should be chosen for numy > 1 .")
    elif GP == "DKL" :
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = DKLRegressionModel(x_train, y_train, likelihood, kernel, x_train.size(-1))
    
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()


    model.train()
    likelihood.train()
    
    #Define the parameters to optimize
    if GP == "DKL":
        optimizer = torch.optim.Adam([
    {'params': model.feature_extractor.parameters()},
    {'params': model.covar_module.parameters()},
    {'params': model.mean_module.parameters()},
    {'params': model.likelihood.parameters()},
], lr=0.01)
    else:
        optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        ], lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    #training_iter = 500
    
    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(x_train)
        loss = -mll(output, y_train)
        loss.backward()
        if verbose:
            if i % 100  == 0 or i == n_iter - 1:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, n_iter, loss.item()))
        optimizer.step()

    mse, r2, predictions = InferGPModel(likelihood, model, x_test, y_test)
    gpnoise = estimate_gpnoise(likelihood, verbose=True)  
  
    return mse, r2, gpnoise ,predictions, likelihood, model


def InferGPModel(likelihood, model, Xinfer, YTrue, scaleY=None):
    """
    Inference on optimized models. Appied in two settings a) Application over AL query window b) Validating a test set and scaling to normal units. 
    Input Args:
    likelihood : Optimized GPyTorch likelihood object
    model      : Optimized GPyTorch object
    Xinfer     : Feature to predict 
    YTrue      : Ground truth 
    scaleY     : SKlearn scaler object to be used with test set
    Return:
    rmse, r2, predictions: float, float, tensor
    """ 
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np 

    if type(Xinfer) == np.ndarray:
        use_cuda = torch.cuda.is_available()
        dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        Xinfer = torch.from_numpy(Xinfer).type(dtype)    
        YTrue = torch.from_numpy(YTrue).type(dtype)

    # Set into eval mode
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihood(model(Xinfer))
        mean = predictions.mean
        lower, upper = predictions.confidence_region()
    if scaleY != None:
        if torch.cuda.is_available():
            mean = scaleY.inverse_transform(predictions.mean.cpu() )
            YTrue = scaleY.inverse_transform(YTrue.cpu() )
        else:
            mean = scaleY.inverse_transform(predictions.mean.detach().numpy())      
            YTrue = scaleY.inverse_transform(YTrue)
    else :
        mean = mean.cpu() 
        YTrue = YTrue.cpu()

    mse = mean_squared_error(YTrue,mean)
    r2 = r2_score(YTrue,mean)

    return np.sqrt(mse), r2, predictions


def estimate_gpnoise(likelihood, verbose=True):
    """
    Estimate true GP Noise under constraints
    Input Args:
    likelihood : Optimized GPyTorch likelihood object
    verbose : default = True (bool)
    Return:
    GPnoise : 1x1 tensor 
    """
    import torch
    
    raw_noise = likelihood.noise_covar.raw_noise   
    constraint = likelihood.noise_covar.raw_noise_constraint
    outputnoise = constraint.transform(raw_noise)
    if verbose :
        print(f'Actual noise: {outputnoise.item()}')

    return  torch.tensor( outputnoise.item() )

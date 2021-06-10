"""
@author: gsivaraman@anl.gov
"""
from copy import deepcopy

class OTFSampler:
    def __init__(self, index):
        """Different On-The-Fly (OTF) samplers are implemented here"""
    
    
        """
        Input Args:
        index : index list (list)
        """
        self.index = deepcopy(index)
        self.index_otf = []
        
    def _randsampler(self,width):
        """
        OTF sampling  using random sampling
        Input Args:
        width : sampling width (int)
        Return:
        index : A list of (index original - sample picked for training)
        ind_reduced : Appended index for learning (list)
        """
        import random
        lsample = random.sample(self.index, width)
            
        for s in lsample : 
            self.index.remove(s)
            self.index_otf.append(s)
            
        return self.index, self.index_otf 
    

    def _initsampler(self,width):
        """
        OTF sampling  using initial sampling
        Input Args:
        width : sampling width (int)
        Return:
        index : A list of (index original - sample picked for training)
        ind_reduced : Appended index for learning (list)
        """
        lsample = range(width)
            
        for s in lsample : 
            self.index.remove(s)
            self.index_otf.append(s)
            
        return self.index, self.index_otf 
    
    
    def _slidingwindowsampler(self,  window, slide):
        """
        OTF sampling  using sliding window sampler
        %GP uncertainity Q_${unc}$(u) = armin{xi elem u} \frac{|mu(xi)|}{var(xi)}
        Input Args:
        window : Sampling window (int)
        slide : sampling slide (int)
        Return:
        iterator
        """
        try: it = iter(self.index)
        except TypeError:
            raise Exception("**ERROR** sequence must be iterable.")
        if not ((type(window) == type(0)) and (type(slide) == type(0))):
            raise Exception("**ERROR** type(window) and type(slide) must be int.")
        if slide > window:
            raise Exception("**ERROR** slide must not be larger than sampling window.")
        if window > len(self.index):
            raise Exception("**ERROR** window must not be larger than sample length.")
        
        numOfChunks = int( ((len(self.index)- window)/slide)+1 )
        for i in range(0,numOfChunks*slide,slide):
            yield self.index[i:i+window]
            



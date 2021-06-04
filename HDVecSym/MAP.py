import numpy as np



def rotate(HDvectors,rotateby=1):
    return np.roll(HDvectors, rotateby, axis = 0)
    
def bind(HDvectors1,HDvectors2):
    if HDvectors1.shape == HDvectors2.shape:
        return HDvectors1*HDvectors2
    elif len(HDvectors1.shape)==1 or len(HDvectors2.shape)==1 or HDvectors1.shape[1]==1 or HDvectors2.shape[1]==1:    
        if len(HDvectors1.shape)==1 or HDvectors1.shape[1]==1:
            return HDvectors2*np.tile(HDvectors1, (1, HDvectors2.shape[1]))   
        elif len(HDvectors2.shape)==1 or HDvectors2.shape[1]==1:
            return HDvectors1*np.tile(HDvectors2, (1, HDvectors1.shape[1]))          
    else:
        raise Exception("Dimensions of arrays must agree or one of them should be a vector")             
    

def unbind(HDvectors1,HDvectors2):
    if HDvectors1.shape == HDvectors2.shape:
        return HDvectors1*HDvectors2                
    else:
        raise Exception("Dimensions of arrays must agree or one of them should be a vector")        
    
def bundle(HDvectors, optype="unrestricted", kappa=3):
    if len(HDvectors[0].shape) == 1:
        base = HDvectors[0] 
    else:        
        base = np.sum(HDvectors[0], axis=1)    
        base = np.expand_dims(base, axis=1) #to keep (N,1) shape
       
    #Regular addition
    if optype=="unrestricted":
        return base
    #Clipping function
    elif optype=="clipping":
        base[base>kappa]=kappa
        base[base<-kappa]=-kappa        
        return base
    else:
        raise Exception("The type is not defined")        



       
    
def item(concepts,  N=1000):
    memory = 2*np.random.randint(low = 0, high = 2, size =(N, len(concepts))) -1 
    return [memory,concepts]



def similarity(HDvectors1,HDvectors2,stype="dot"):
    dp=np.dot(np.transpose(HDvectors1),HDvectors2) 
    if stype=="dot":
            return dp
    elif stype=="cosine": 
        norms=(np.dot( np.expand_dims( np.linalg.norm(HDvectors1,axis=0), axis=1),  np.expand_dims(np.linalg.norm(HDvectors2,axis=0), axis=0)))
        dp.astype(float)
        return np.divide(dp,norms)  
    else:
        raise Exception("The type is not defined")       

def getitems(itemmem, concepts):
    HDvectors=np.zeros((itemmem[0].shape[0], len(concepts)), dtype=int)
    for i in range(len(concepts)):
        con=concepts[i]
        ind= itemmem[1].index(con)
        if ind!=-1:         
            HDvectors[:,i]=itemmem[0][:,ind]    
        else:
            raise Exception("The concept is not present in the item memmory") 
    return [HDvectors, concepts]
    

def probe(itemmem, query, searchtype="nearest", simtype="dot"):
    scores=similarity(itemmem[0],query,stype=simtype)
    
    if searchtype=="nearest":
        ind=np.argmax(scores)
        return [ np.expand_dims( itemmem[0][:,ind], axis=1), itemmem[1][ind]]
    else:
        raise Exception("The specified search type is not defined")       
        
    
    


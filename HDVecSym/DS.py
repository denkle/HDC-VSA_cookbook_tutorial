import importlib
import numpy as np


#Representation of sets
def sethd(itemmem, concepts, vsatype="MAP", bundlingtype="unrestricted", kappa=3):
    full_module_name = "HDVecSym." + vsatype #VSAs framework type
    vsamodule = importlib.import_module(full_module_name) # import module
    HDvectors=vsamodule.getitems(itemmem, concepts) # get the corresponding items
    return vsamodule.bundle(HDvectors, bundlingtype, kappa) # perform bundling of the desired HD vectors

#Representation of sequences
def sequence(itemmem, concepts, vsatype="MAP", rerptype="bundling", bundlingtype="unrestricted", kappa=3):
    full_module_name = "HDVecSym." + vsatype #VSAs framework type
    vsamodule = importlib.import_module(full_module_name) # import module
    HDvectors=vsamodule.getitems(itemmem, concepts)
    
    #prepare rotated HD vectors
    for i in range(1,len(concepts)):
        HDvectors[0][:,i] = vsamodule.rotate(HDvectors[0][:,i], rotateby=i)
    
    #create composite HD vector
    if rerptype=="bundling": # if the representation is bundling based
        return vsamodule.bundle(HDvectors, bundlingtype, kappa) # perform bundling of the desired HD vectors            
    elif rerptype=="binding": # if the representation is binding based
        HDseq=HDvectors[0][:,0]
        for i in range(1,len(concepts)):
            HDseq=vsamodule.bind(HDseq, HDvectors[0][:,i])
        return np.expand_dims(HDseq, axis=1) #to keep (N,1) shape

#edges=[("a","b"),("a","e"),("c","b"),("d","c"),("e","d")]
#Representation of graphs
def graph(itemmem, edges, vsatype="MAP", graphtype="undirected", bundlingtype="unrestricted", kappa=3):
    full_module_name = "HDVecSym." + vsatype #VSAs framework type
    vsamodule = importlib.import_module(full_module_name) # import module 
    #HDgraph=np.zeros((itemmem[0].shape[0], 1), dtype=itemmem[0].dtype)    
    HDedges=np.zeros((itemmem[0].shape[0], len(edges)), dtype=itemmem[0].dtype)
    
    for i in range(len(edges)):
        vert1=edges[i][0]
        vert2=edges[i][1]
        HDvert1=vsamodule.getitems(itemmem, [vert1])
        HDvert2=vsamodule.getitems(itemmem, [vert2])
        if graphtype=="undirected":
            HDedges[:,i]=np.squeeze(vsamodule.bind(HDvert1[0], HDvert2[0]),axis=1)    
        elif graphtype=="directed":
            HDedges[:,i]=np.squeeze(vsamodule.bind(HDvert1[0], vsamodule.rotate(HDvert2[0], rotateby=1)),axis=1)
    
    return vsamodule.bundle([HDedges], bundlingtype, kappa)     
            
#transitions=[["L","L","P"],["L","U","T"],["U","U","T"],["U","L","P"]]
#Representation of finite state automata
def fsa(itemmemstates, itemmeminput, transitions, vsatype="MAP", bundlingtype="unrestricted", kappa=3):
    full_module_name = "HDVecSym." + vsatype #VSAs framework type
    vsamodule = importlib.import_module(full_module_name) # import module     
    HDtransitions=np.zeros((itemmemstates[0].shape[0], len(transitions)), dtype=itemmemstates[0].dtype)
     
    for i in range(len(transitions)):
        HDstates=sequence(itemmemstates, transitions[i][0:2], vsatype=vsatype, rerptype="binding") 
        HDinput=vsamodule.getitems(itemmeminput, [transitions[i][2]])
        HDtransitions[:,i]=np.squeeze(vsamodule.bind(HDstates, HDinput[0]),axis=1) 
  
    return vsamodule.bundle([HDtransitions], bundlingtype, kappa)  


#Representation of frequency distributions
def frequncy(itemmem, concepts, frequency, vsatype="MAP", bundlingtype="unrestricted", kappa=3):
    full_module_name = "HDVecSym." + vsatype #VSAs framework type
    vsamodule = importlib.import_module(full_module_name) # import module
    HDvectors=vsamodule.getitems(itemmem, concepts) # get the corresponding items
    HDvectors_scaled=np.zeros((HDvectors[0].shape))
    for i in range(len(frequency)):
        HDvectors_scaled[:,i]=frequency[i]*HDvectors[0][:,i]    
    return vsamodule.bundle([HDvectors_scaled], bundlingtype, kappa) # perform bundling of the desired HD vectors


#Representation of n-gram statistics
def ngram(itemmem, data, n, vsatype="MAP", bundlingtype="unrestricted", kappa=3):
    full_module_name = "HDVecSym." + vsatype #VSAs framework type
    vsamodule = importlib.import_module(full_module_name) # import module
    
    HDgram=np.zeros((itemmem[0].shape[0], 1), dtype=itemmem[0].dtype) # will store statistics 
    #Go via all n-grams
    for i in range(len(data)-n+1):
        HDseq=sequence(itemmem, data[i:i+n], vsatype=vsatype, rerptype="binding")  # get the representation of n-gram
        HDgram+=HDseq # increment statistcs
    return vsamodule.bundle([HDgram], bundlingtype, kappa) # perform desired bundling on the accummulating HD vector


#Representation of trees
def tree(itemroles, itemsymbols, treelist, vsatype="MAP", bundlingtype="unrestricted", kappa=3):
    full_module_name = "HDVecSym." + vsatype #VSAs framework type
    vsamodule = importlib.import_module(full_module_name) # import module     
    HDtransitions=np.zeros((itemroles[0].shape[0], len(treelist)), dtype=itemroles[0].dtype)
     
    for i in range(len(treelist)):
        HDpath=sequence(itemroles, treelist[i][1], vsatype=vsatype, rerptype="binding") 
        HDsym=vsamodule.getitems(itemsymbols, [treelist[i][0]])
        HDtransitions[:,i]=np.squeeze(vsamodule.bind(HDpath, HDsym[0]),axis=1) 
  
    return vsamodule.bundle([HDtransitions], bundlingtype, kappa)  




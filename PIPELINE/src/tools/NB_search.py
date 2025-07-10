import faiss
from sklearn.preprocessing import normalize
import numpy as np

def neighborhood_search(emb,thresh,k_neighbors):

    index = faiss.IndexFlatIP(emb.shape[1])
    faiss.normalize_L2(emb)
    index.add(emb)
    
    sim, I = index.search(emb, k_neighbors)
    
    pred_index=[]
    pred_sim=[]
    for i in range(emb.shape[0]):
        cut_index=0
        for j in sim[i]:
            if(j>thresh):
                cut_index+=1
            else:
                break
              
        pred_index .append( I[i][:(cut_index)])
        pred_sim .append (sim[i][:(cut_index)])
       
    return pred_index,pred_sim
    

def blend_neighborhood(emb, match_index_lst, similarities_lst):
    new_emb = emb.copy()
    for i in range(emb.shape[0]):
        cur_emb = emb[match_index_lst[i]]
        
        weights = np.expand_dims(similarities_lst[i], 1)
        
        new_emb[i] = (cur_emb * weights).sum(axis=0)
        
    new_emb = normalize(new_emb, axis=1)
    
    return new_emb

def iterative_neighborhood_blending(emb, threshes,k_neighbors):
    for thresh in threshes:
        match_index_lst, similarities_lst = neighborhood_search(emb, thresh,k_neighbors)
        emb = blend_neighborhood(emb, match_index_lst, similarities_lst)
    return emb,match_index_lst,similarities_lst
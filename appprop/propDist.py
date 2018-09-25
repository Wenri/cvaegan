from appProp import approxProp,fullProp
from scipy.spatial.distance import pdist,cdist,squareform
import numpy as np

def approxDist(feaA, feaB, attrs, l=0.2):
	Apd=pdist(feaA, 'euclidean')
	A=squareform(np.exp(-Apd/l))
	Bpd=cdist(feaB, feaA, 'euclidean' )
	B=np.exp(-Bpd/l)
	w=np.zeros(attrs.shape[0])
	w[:feaA.shape[0]] = 1
	return np.concatenate(
			[ approxProp(A,B, g=g, w=w) for g in attrs.T ],
			axis=1
		)

def fullDist(fea, attrs, l=0.2, semiratio=0.2):
	Zpd=pdist(fea, 'euclidean')
	Z=squareform(np.exp(-Zpd/l))
	w=np.zeros(attrs.shape[0])
	w[:int(attrs.shape[0]*semiratio)] = 1
	return np.concatenate(
			[ fullProp(Z=Z, g=g, w=w) for g in attrs.T ],
			axis=1
		)

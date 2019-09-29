import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt

from scipy.io import loadmat
from tensorly.decomposition import parafac, tucker
from itertools import combinations_with_replacement


def extract_tensor(mat, key):
	return tl.tensor(mat[key][0][0][0])


def calc_AIC(mat, factors):
	recon = tl.kruskal_to_tensor(factors)
	

def rank_search_parafac(tensor, rank_range):
	AIC = []
	for rank in range(1,rank_range+1):
		factors = parafac(tensor, rank=rank)
		recon = tl.kruskal_to_tensor(factors)
		err = tensor - recon
		rank_AIC = 2 * tl.tenalg.inner(err, err) + 2 * rank
		AIC.append(rank_AIC)
		
	return AIC


def rank_search_tucker(tensor, rank_range):
	AIC = {}
	for rank in combinations_with_replacement(range(1, rank_range + 1), 3):
		decomp = tucker(tensor, rank)
		recon = tl.tucker_to_tensor(decomp)
		err = tensor - recon
		rank_AIC = 2 * tl.tenalg.inner(err, err) + 2 * sum(rank)
		AIC[rank] = rank_AIC
		
	return AIC


def get_min_key(d):
	return min(d, key=d.get)


def build_xy_plots_cp(decomp):
	XY = []
	weights = decomp[0]
	factors = decomp[1]
	for i in range(decomp.rank):
		new_xy = tl.tenalg.kronecker([factors[0][:,i], factors[1][:,i].reshape(-1,1)])*weights[i]
		XY.append(new_xy)
		
	return XY
	

def build_xy_plots_tucker(decomp):
	XY = []
	factors = decomp[1]
	for i in range(factors[0].shape[1]):
		new_xy = tl.tenalg.kronecker([factors[0][:,i], factors[1][:,i].reshape(-1,1)])
		XY.append(new_xy)
		
	return XY
	
	
def show_plots(plots, rows=2):
	cols = np.ceil(len(plots) / rows)
	fig, axes = plt.subplots(int(rows), int(cols))
	row = 1
	for i in range(len(plots)):
		col_idx = int(i % cols)
		if (i / cols) != 1:
			axes[row-1, col_idx].matshow(plots[i])
		else:
			row += 1
			axes[row-1, col_idx].matshow(plots[i])
	plt.show()
	

if __name__ == "__main__":

	# Read in tensors and convert to numpy format
	tensors = loadmat('heatT.mat')
	tensor_names = [key for key in tensors.keys() if not '__' in key]
	np_tensors = [extract_tensor(tensors, name) for name in tensor_names]

	# For each tensor, do a decomposition and find which rank gives the minimum AIC
	best_cp_aic = [np.argmin(rank_search_parafac(tensor, 10)) for tensor in np_tensors]
	best_tucker_aic = [get_min_key(rank_search_tucker(tensor, 5)) for tensor in np_tensors]
	
	# Decompose the tensors using the optimal rank we found above
	cp_decomps = [parafac(np_tensors[idx], rank=best_cp_aic[idx]) for idx in range(len(np_tensors))]
	tucker_decomps = [tucker(np_tensors[idx], rank=best_tucker_aic[idx]) for idx in range(len(np_tensors))]
	
	# Plot the spatial and temporal patterns in an attempt to visually distinguish between the two materials
	for decomp in cp_decomps:
		plots = build_xy_plots_cp(decomp)
		show_plots(plots)
		plt.plot(decomp[1][2])
		plt.show()

	for decomp in tucker_decomps:
		plots = build_xy_plots_tucker(decomp)
		show_plots(plots)
		plt.plot(decomp[1][2])
		plt.show()

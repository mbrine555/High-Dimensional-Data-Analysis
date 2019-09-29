import numpy as np
from PIL import Image
from scipy.signal import correlate
import matplotlib.pyplot as plt


def compute_gradient(image, i, j):
	partial_x = compute_x_deriv(image, i, j)
	partial_y = compute_y_deriv(image, i, j)
	
	mag = np.sqrt(partial_x**2 + partial_y**2)
	direction = np.arctan(partial_x / partial_y)
	return mag, direction
	

def compute_x_deriv(image, i, j):
	ds_dx = (1/2) * (image[i+1, j] - image[i, j] + image[i+1, j+1] - image[i, j+1])
	return ds_dx


def compute_y_deriv(image, i, j):
	ds_dy = (1/2) * (image[i, j+1] - image[i, j] + image[i+1, j+1] - image[i+1, j])
	return ds_dy

	
def get_neighbor_pixels(direction, i, j):
	if (-(1/8) * np.pi) < direction[i, j] <= ((1/8) * np.pi):
		return ([i, j-1], [i, j+1])
	
	if ((1/8) * np.pi) < direction[i, j] <= ((3/8) * np.pi):
		return ([i+1, j-1], [i-1, j+1])
		
	if (-(3/8) * np.pi) < direction[i, j] <= (-(1/8) * np.pi):
		return ([i-1, j-1], [i+1, j+1])
		
	return ([i-1, j], [i+1, j])


def build_intensity_matrix(direction, magnitude):
	phi = np.empty(direction.shape)
	
	for i in range(direction.shape[0]):
		for j in range(direction.shape[1]):
			point = magnitude[i, j]
			neighbors = get_neighbor_pixels(direction, i, j)
			try:
				phi[i, j] = magnitude[i, j] * int((point > magnitude[neighbors[0][0], neighbors[0][1]]) & (point > magnitude[neighbors[1][0], neighbors[1][1]]))
			except:
				phi[i, j] = 0
	return phi


def threshold_hystersis(phi, t_1, t_2):
	edge = np.zeros(np.shape(phi))
	
	count = 1
	while count != 0:
		count = 0
		for i in range(phi.shape[0] - 1):
			for j in range(phi.shape[1] - 1):
				if (phi[i,j] >= t_2) and (edge[i,j] == 0):
					edge[i,j] = 1
					count += 1
				elif (phi[i,j] >= t_1) and (edge[i,j] == 0):
					n = get_neighborhood(i,j)
					for pixel in n:
						if edge[pixel[0], pixel[1]] == 1:
							edge[i, j] = 1
							count += 1
							
	return edge
					
		
def get_neighborhood(i, j):
	n = [[k,l] for k in range(i-1, i+2) for l in range(j-1, j+2) if [k,l] != [i,j]]
	return n	    


if __name__ == '__main__':
	im = Image.open('horse1.jpg').convert('L')
	im = np.array(im)
	
	gaussian_kernel = 1/1115 * np.array([[1, 4, 7, 10, 7, 4, 1],
										[4, 12, 26, 33, 26, 12, 4],
										[7, 26, 55, 71, 55, 26, 7],
										[10, 33, 71, 91, 71, 33, 10],
										[7, 26, 55, 71, 55, 26, 7],
										[4, 12, 26, 33, 26, 12, 4],
										[1, 4, 7, 10, 7, 4, 1]])
	
	filtered_im = correlate(im, gaussian_kernel).round()
	smooth_im = np.clip(filtered_im[3:-3, 3:-3], 0, 255)
	
	smooth_im = smooth_im.astype('int16')
	
	magnitude = np.empty(smooth_im.shape)
	direction = np.empty(smooth_im.shape)
	
	for i in range(smooth_im.shape[0] - 1):
		for j in range(smooth_im.shape[1] - 1):
			magnitude[i, j], direction[i, j] = compute_gradient(smooth_im, i, j)
	
	phi = build_intensity_matrix(direction, magnitude)
	edge_image = threshold_hystersis(phi, 3, 8)
	plt.imshow(edge_image, cmap=plt.cm.gray)
	plt.show()
	
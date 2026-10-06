import numpy as np

def pos_encoding(position: int, d_model: int):
	
	angle_rads = np.arange(position)[:,np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model)
	)

	angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
	angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

	#  pos_encoding = angle_rads[np.newaxis, ...] # Batch dimension: 
	# Before: (position, d_model) -> (1, position, d_model)

	# pos_encoding = np.float16(pos_encoding)
	pos_encoding = np.float16(angle_rads)

	return pos_encoding
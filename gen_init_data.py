import numpy as np 
import torch  

rand_noise = torch.randn([1, 80,  300])
np.savetxt("rand_noise_1_80_300.txt", rand_noise.numpy().reshape(-1), delimiter=",")

speech_window = np.hamming(2 * 8 * 480)
np.savetxt("speech_window_2x8x480.txt", speech_window.reshape(-1), delimiter=",")
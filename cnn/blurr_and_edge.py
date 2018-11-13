#Gaussian blurring and Edge detection
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('lena.png')
plt.imshow(img)
plt.show()

bw = img.mean(axis=2)
plt.imshow(bw, cmap='gray')
plt.show()

W = np.zeros((20,20))
for i in xrange(20):
    for j in xrange(20):
        dist = (i-9.5)**2 + (j-9.5)**2
        W[i,j] = np.exp(-dist/50)
W/=W.sum()
plt.imshow(W,cmap='gray')
plt.show()

out = convolve2d(bw, W, mode='same')
plt.imshow(out,cmap='gray')
plt.show()

Hx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=np.float32) #horizontal edge
Hy = Hx.T #vertical edge
Gx = convolve2d(bw,Hx,mode='same')
plt.imshow(Gx, cmap='gray')
plt.show()
Gy = convolve2d(bw,Hy,mode='same')
plt.imshow(Gy, cmap='gray')
plt.show()
G = np.sqrt(Gx*Gx + Gy*Gy)
plt.imshow(G, cmap='gray')
plt.show()
theta = np.arctan2(Gy,Gx)
plt.imshow(theta, cmap = 'gray')
plt.show()

import numpy as np
import skimage
from scipy import ndimage as ndi
from skimage import io
from skimage.color import rgb2gray
from skimage.morphology import disk
from skimage.filters.rank import gradient
from skimage.segmentation import watershed
from skimage.filters import rank
from sklearn.cluster import KMeans
image = io.imread('4.tif')
image = image[:,95:1500,:]
image = rgb2gray(image)

denoised = rank.median(image, disk(5))

markers = rank.gradient(denoised, disk(5)) < 10
markers = ndi.label(markers)[0]


gradient_image = gradient(denoised, disk(5))
X = gradient_image.reshape(-1,1)
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
labels = kmeans.predict(X)
labels = labels.reshape(image.shape)
segment = labels*(255/3)
segmented_image = watershed(gradient_image,markers)
io.imsave('4test.png',segment)

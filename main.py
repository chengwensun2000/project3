import skimage.io
import skimage.filters
import skimage.segmentation
import skimage.morphology
import skimage.color
import cv2
from skimage.morphology import disk
from skimage.filters.rank import gradient
from skimage.segmentation import watershed
import numpy as np
# Load the image
image = cv2.imread('4.tif')
image = image[:,95:1500,:]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Pre-process the image to enhance the features
processed_image = skimage.filters.gaussian(gray, sigma=1)

stucturing_elment = disk(15)
gradient_image = gradient(processed_image,stucturing_elment)
segmentend_image = watershed(gradient_image,gray)
cv2.imshow('here',gradient_image)
cv2.waitKey(0)
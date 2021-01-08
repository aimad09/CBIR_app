from skimage import feature
import numpy as np
import cv2
class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# store the number of points and radius
		self.numPoints = numPoints
		self.radius = radius

	def describe(self, image, eps=1e-7):
            #convert image to grayscale image
            grayImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            #get the features
            lbp = feature.local_binary_pattern(grayImage, self.numPoints,self.radius, method="uniform")
            (hist,_) = np.histogram(lbp.ravel(),bins=np.arange(0, self.numPoints + 3),range=(0, self.numPoints + 2))

            #normalize the histograme
            hist = hist.astype("float")
            hist /= (hist.sum()+eps)
            
            return hist.flatten()
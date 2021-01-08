																																																																																																																																																																																										
import imutils
import cv2
class RGBHistogram:
	def __init__(self, bins):
		# store the number of bins the histogram will use
		self.bins = bins

	def describe(self, image):
		# compute a 3D histogram in the RGB colorspace,
		# then normalize it to make sure two same images with diffrent scale will have the same feature vectors
		hist = cv2.calcHist([image], [0, 1, 2],
			None, self.bins, [0, 256, 0, 256, 0, 256])

		# normalize with OpenCV 2.4
		if imutils.is_cv2():
			hist = cv2.normalize(hist)
		# otherwise normalize with OpenCV 3+
		else:
			hist = cv2.normalize(hist,hist)
		# return out 3D histogram as a features vector
		return hist.flatten()
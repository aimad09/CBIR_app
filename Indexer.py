from colorDescriptor import RGBHistogram
from LBP import LocalBinaryPatterns
from imutils.paths import list_images
import pickle
import cv2
import os

# method for writing the index into disk
def write_index(file,index):																																		
	f = open(file, "wb")									
	f.write(pickle.dumps(index))	
	f.close()
	print("[INFO] done...indexed {} images".format(len(index)))
	return index		


DATA_PATH = "./static/data"

# initialize the index dictionaries to store our quantifed
# images, with the 'key' of the dictionary being the image
# filename and the 'value' our computed features
colorIndex = {}
textureIndex = {}

# initialize our image descriptors  a 3D RGB histogram with and  loacal binary Pattern 
# 8 bins per channel
colorDesc = RGBHistogram([8, 8, 8])
#24 the number of points and 8 for the radius
textureDesc =LocalBinaryPatterns(24,8) 

# use list_images to grab the image paths and loop over them
#each category we grab about 40 images 
for category in os.listdir(DATA_PATH):
	i=0
	for imagePath in list_images(os.path.join(DATA_PATH,category)) :
		# extract our unique image ID (i.e. the /category/filename)
		k = imagePath[imagePath.rfind("/data") + 6:]
		# load the image, describe it using our RGB histogram
		# descriptor, and update the index
		image = cv2.imread(imagePath)
		colorFeatures = colorDesc.describe(image)
		textureFeatures =textureDesc.describe(image)
		colorIndex[k] = colorFeatures
		textureIndex[k] = textureFeatures
		print(k)
		if i >=40 :
			break
		i=i+1

# we are now done indexing our image, now we can write our
# index to disk
write_index("colorIndex",colorIndex)
write_index("textureIndex",textureIndex)


																																															
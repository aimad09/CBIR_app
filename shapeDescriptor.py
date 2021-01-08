import cv2

class Hu_Moments : 

    def describe(self,image) :
        gray_image = cv2.cvtColor(image,cv2.BGR2GRAY)
        return cv2.HuMoments(cv2.moments(gray_image)).flatten()
import cv2 as cv
import numpy as np
from skimage.feature import local_binary_pattern as lbp

from histo import histMean
from histo import histDeviation

nbPart = 500
# Initiate ORB detector
# brief = cv.DescriptorExtractor_create("BRIEF")
# brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
# orb = cv.ORB_create(nfeatures=200000, scoreType=cv.ORB_FAST_SCORE)
# fast = cv.FastFeatureDetector_create()

def focus(event, x, y, flags,  env):
	if event == cv.EVENT_MOUSEMOVE and env[6] is None:
		env[0], env[1] = x, y
	if event == cv.EVENT_FLAG_LBUTTON and env[5] is None:
		env[0], env[1] = x, y
		# roi = env[4][env[1]-10:env[1]+10, env[0]-10:env[0]+10, :]
		roi = env[4][env[1]-env[3]+1:env[1]+env[3], env[0]-env[2]+1:env[0]+env[2], :]
		nbPix = (env[2] * 2 - 1)*(env[3] * 2 - 1)
		cv.imshow("histSelected", roi)
		roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
		# env[5] = []
		hist = cv.calcHist([roi], [0, 1], None, [20, 20], [0, 180, 0, 256])
		print(hist.shape)
		env[5] = [hist]

		lbpH = lbp(roi[:,:,0], 3 * 8, 3 , method='uniform')
		lbpS = lbp(roi[:,:,1], 3 * 8, 3 , method='uniform')
		lbpTot = np.ones([lbpH.shape[0],lbpH.shape[1],2])
		lbpTot[:, :, 0] = lbpH
		lbpTot[:, :, 1] = lbpS
		print(lbpTot.shape)
		hist = cv.calcHist([lbpTot.astype(np.float32)], [0,1], None, [20,20], [0, 255, 0, 255])
		env[5].append(hist)


		# find the keypoints with ORB
		# kp = orb.detect(roi,None)
		# kp = fast.detect(roi,None)
		# compute the descriptors with ORB
		# print(kp)
		# kp, des = brief.compute(roi, kp)
		# kp, des = orb.compute(roi, kp)
		# kp, des = fast.compute(roi, kp)
		# print("orb ref model")
		# print(des)
		# print(kp)
		# env[5].append(des)


		# mu = histMean(hist, nbPix)
		# sigma = histDeviation(hist, mu, nbPix)
		# env[5].append([mu,sigma])
		# hist = cv.calcHist([roi], [1], None, [256], [0,256])
		# mu = histMean(hist, nbPix)
		# sigma = histDeviation(hist, mu, nbPix)
		# env[5].append([mu,sigma])
	if event == cv.EVENT_FLAG_RBUTTON and env[5] is not None and env[6] is None:
		env[6] = (1 * np.random.rand(nbPart, 3)).astype(np.float32)
		env[6][...,0] *= env[2] * 2 - 1
		env[6][...,0] += env[0] - env[2] + 1
		env[6][...,1] *= env[3] * 2 - 1
		env[6][...,1] += env[1] - env[3] + 1

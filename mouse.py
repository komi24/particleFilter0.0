import cv2 as cv
import numpy as np

from histo import histMean
from histo import histDeviation

nbPart = 500

def focus(event, x, y, flags,  env):
	if event == cv.EVENT_MOUSEMOVE and env[6] is None:
		env[0], env[1] = x, y
	if event == cv.EVENT_FLAG_LBUTTON and env[5] is None:
		env[0], env[1] = x, y
		roi = env[4][env[1]-env[3]+1:env[1]+env[3], env[0]-env[2]+1:env[0]+env[2]]
		nbPix = (env[2] * 2 - 1)*(env[3] * 2 - 1)
		cv.imshow("histSelected", roi)
		roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
		env[5] = []
		hist = cv.calcHist([roi], [0], None, [180], [0,180])
		mu = histMean(hist, nbPix)
		sigma = histDeviation(hist, mu, nbPix)
		env[5].append([mu,sigma])
		hist = cv.calcHist([roi], [1], None, [256], [0,256])
		mu = histMean(hist, nbPix)
		sigma = histDeviation(hist, mu, nbPix)
		env[5].append([mu,sigma])
	if event == cv.EVENT_FLAG_RBUTTON and env[5] is not None and env[6] is None:
		env[6] = (np.random.rand(nbPart, 3)).astype(np.float32)
		env[6][...,0] *= env[2] * 2 - 1
		env[6][...,0] += env[0] - env[2] + 1
		env[6][...,1] *= env[3] * 2 - 1
		env[6][...,1] += env[1] - env[3] + 1

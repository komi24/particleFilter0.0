import cv2 as cv
import math as m

def histMean(hist, nbPix):
	ret = 0
	for i in range(len(hist)):
		ret += i * hist[i][0]
	ret /= nbPix
	return ret

def histDeviation(hist, mean, nbPix):
	ret = 0
	for i in range(len(hist)):
		ret += pow(i - mean, 2) * hist[i][0]
	ret /= nbPix
	return m.sqrt(ret)


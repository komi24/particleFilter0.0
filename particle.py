import cv2 as cv
from skimage.feature import local_binary_pattern as lbp
import numpy as np
import math as m
from random import random as r
from random import gauss as g

# brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
# orb = cv.ORB_create(nfeatures=200000)
# fast = cv.FastFeatureDetector_create()

def moveP(prvs, nxt, env):
	a,b,c,d = int(env[1]-env[3]),int(env[1]+env[3]),int(env[0]-env[2]),int(env[0]+env[2])
	flow = cv.calcOpticalFlowFarneback(prvs[a:b,c:d],nxt[a:b,c:d], None, 0.5, 3, 15, 3, 5, 1.2, 0)
	dx = np.mean(flow[...,0])
	stdx = np.std(flow[...,0])
	dy = np.mean(flow[...,1])
	stdy = np.std(flow[...,1])
	bestPIndex = np.argmax(env[6][...,2])
	print(env[6].shape)
	bestPartcles = np.mean(env[6][env[6][:,1].argsort()], axis=0)
	# bestPartcles = np.mean(np.sort(env[6].view('i8,i8,i8'), order=['f2'], axis=0)[-10 : ], axis=[0,1])
	print(bestPartcles)
	for i in range(len(env[6])):
		env[6][i][0] += 1.5 * dx + 0.5 * (bestPartcles[0] - env[6][i, 0]) + 2.1 * g(0, 1)
		# env[6][i][0] += 0.2 * dx + 0.5 * (env[6][bestPIndex,0] - env[6][i, 0]) + 2.1 * g(0, 1)
		# env[6][i][0] += 1.2 * dx + 0.4 * g(0, 1)
		# env[6][i][1] += 1.2 * dy + 0.4 * g(0, 1)
		# env[6][i][1] += 0.2 * dy + 0.5 * (env[6][bestPIndex,1] - env[6][i, 1]) + 2.1 * g(0, 1)
		env[6][i][1] += 1.5 * dy + 0.5 * (bestPartcles[1] - env[6][i, 1]) + 2.1 * g(0, 1)
		# env[6][i][0] += g(dx, stdx)
		# env[6][i][1] += g(dy, stdy)
		

def gaussian(x,mu,sigma):
	# return 1/(sigma * m.sqrt(2*m.pi))*m.exp(-pow(x-mu,2)/(2*pow(sigma,2)))
	return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))

def gaussianWeight(env):
	hsv = cv.cvtColor(env[4], cv.COLOR_BGR2HSV)
	i = 0
	
	# kp, des = fast.compute(env[4], [ cv.KeyPoint(i[0],i[1], 1.0) for i in env[6]])
	# kp, des = brief.compute(env[4], [ cv.KeyPoint(i[0],i[1], 1.0) for i in env[6]])
	# kp, des = orb.compute(env[4], [ cv.KeyPoint(i[0],i[1], 1.0) for i in env[6]])
	# print(des.shape)
	for p in env[6]:
		x,y,w = p.ravel()
		x,y = int(x), int(y)
		w = 1.0
		# Compute the weight according to color histograms
		roi = hsv[int(env[1]-env[3]+1):int(env[1]+env[3]), int(env[0]-env[2])+1:int(env[0]+env[2]), :]
		hist = cv.calcHist([roi], [0, 1], None, [20, 20], [0, 180, 0, 256])
		w, tp, tp2 = cv.EMD(hist + 0.001 * np.ones(hist.shape, np.float32), env[5][0] + 0.001 * np.ones(hist.shape, np.float32), cv.DIST_L2)
		hist_feature = gaussian(w, 0, 2)

		# lbpH = lbp(roi[:,:,0], 3 * 8, 3 , method='uniform')
		# lbpS = lbp(roi[:,:,1], 3 * 8, 3 , method='uniform')
		# lbpTot = np.ones([lbpH.shape[0],lbpH.shape[1],2])
		# lbpTot[:, :, 0] = lbpH
		# lbpTot[:, :, 1] = lbpS
		# # print(lbpTot.shape)
		# hist2 = cv.calcHist([lbpTot.astype(np.float32)], [0,1], None, [20,20], [0, 26, 0, 26])
		# w2, tp, tp2 = cv.EMD(hist2 + 0.001 * np.ones(hist2.shape, np.float32), env[5][1] + 0.001 * np.ones(hist2.shape, np.float32), cv.DIST_L2)
		# lbp_feature = gaussian(w2, 0, 5)

		if env[7] == 'display':
			# cv.imshow("lbpH", cv.resize(lbpH, (lbpH.shape[0] * 10, lbpH.shape[1] * 10)))
			# cv.imshow("lbpS", cv.resize(lbpS, (lbpS.shape[0] * 10, lbpS.shape[1] * 10)))
			cv.imshow("hist", cv.resize(hist, (hist.shape[0] * 10, hist.shape[1] * 10)))
			# print(np.max(lbpH))
			# print(np.min(lbpH))
			# print(np.mean(lbpH))
			# cv.imshow("hist LBP", cv.resize(hist2, (hist2.shape[0] * 10, hist2.shape[1] * 10)))
			cv.waitKey(0)
			env[7] = None
		# orb_feature = gaussian(np.min([np.linalg.norm(refModel - des[i]) for refModel in env[5][1]]), 0, 5)
		# print(orb_feature)
		# for channel in range(2):
			# w *= gaussian(hsv[y][x][channel], env[5][channel][0], env[5][channel][1])
		env[6][i][2] = hist_feature #* lbp_feature #+ orb_feature
		# env[6][i][2] = gaussian(w, 0, 5)
		# print(env[6][i][2])
		i += 1

# REMARQUES IMPORTANTE PAR RAPPORT AU TUTO:
# 1) les donnees de l histo ne sont pas reevaluEes au cours du track
# 2) au lieu des 8 points * 2 coord spatiales (donc 16) on a que 3
#		donnees RGB
# 2bis) sans oublier que les 8 distances du tuto sont des float32 tandis que nos valeurs RGB sont des uint8!
# 3) pas de modelisation de deplacement des particules (opticalFlow...?) 

def resample(env):
	print("norm")
	print(np.linalg.norm(env[6][:,2], 2))
	print(np.linalg.norm(env[6][:,2], 1))

	if(np.linalg.norm(env[6][:,2], 2) > 50000):
		print("resample")
		x = env[6][:,0]
		y = env[6][:,1]
		w = env[6][:,2]
		x += 40.0 * g(0 , 1)
		y += 40.0 * g(0 , 1)
		env[6][:,0] = x
		env[6][:,1] = y
		env[6][:,2] = 1.0 / len(env[6])
		gaussianWeight(env)
		return env[6]
	else:
		n = len(env[6])
		index = int(r() * n)	
		beta = 0.0
		mw = max(env[6][...,2])

		ret = (np.random.rand(n, 3)).astype(np.float32)
		for i in range(n):
			beta += r() * 1.2 * mw

			while beta > env[6][index][2]:
				beta -= env[6][index][2]
				index = (index + 1) % n
			
			ret[i][0] = env[6][index][0]
			ret[i][1] = env[6][index][1]
			ret[i][2] = env[6][index][2]

		return ret

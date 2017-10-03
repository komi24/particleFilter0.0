import cv2 as cv
import numpy as np
import math as m
from random import random as r
from random import gauss as g


def moveP(prvs, nxt, env):
	a,b,c,d = env[1]-env[3],env[1]+env[3],env[0]-env[2],env[0]+env[2]
	flow = cv.calcOpticalFlowFarneback(prvs[a:b,c:d],nxt[a:b,c:d], None, 0.5, 3, 15, 3, 5, 1.2, 0)
	dx = np.mean(flow[...,0])
	stdx = np.std(flow[...,0])
	dy = np.mean(flow[...,1])
	stdy = np.std(flow[...,1])
	for i in range(len(env[6])):
		env[6][i][0] += g(dx, stdx)
		env[6][i][1] += g(dy, stdy)
		

def gaussian(x,mu,sigma):
	return 1/(sigma * m.sqrt(2*m.pi))*m.exp(-pow(x-mu,2)/(2*pow(sigma,2)))

def gaussianWeight(env):
	hsv = cv.cvtColor(env[4], cv.COLOR_BGR2HSV)
	i = 0
	for p in env[6]:
		x,y,w = p.ravel()
		x,y = int(x), int(y)
		w = 1.0
		for channel in range(2):
			w *= gaussian(hsv[y][x][channel], env[5][channel][0], env[5][channel][1])
		env[6][i][2] = w
		i += 1

# REMARQUES IMPORTANTE PAR RAPPORT AU TUTO:
# 1) les donnees de l histo ne sont pas reevaluEes au cours du track
# 2) au lieu des 8 points * 2 coord spatiales (donc 16) on a que 3
#		donnees RGB
# 2bis) sans oublier que les 8 distances du tuto sont des float32 tandis que nos valeurs RGB sont des uint8!
# 3) pas de modelisation de deplacement des particules (opticalFlow...?) 

def resample(env):
	n = len(env[6])
	index = int(r() * n)	
	beta = 0.0
	mw = max(env[6][...,2])

	ret = (np.random.rand(n, 3)).astype(np.float32)
	for i in range(n):
		beta += r() * 2.0 * mw

		while beta > env[6][index][2]:
			beta -= env[6][index][2]
			index = (index + 1) % n
		
		ret[i][0] = env[6][index][0]
		ret[i][1] = env[6][index][1]

	return ret

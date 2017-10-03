import cv2 as cv
from mouse import focus
from keyboard import loop
from particle import moveP
from particle import gaussianWeight
from particle import resample
import time 

winSize = 600
cap = cv.VideoCapture(0)
time.sleep(0.5)
cap.set(3, winSize)
cap.set(4, winSize)
#cap = cv.VideoCapture("foot.mp4")


#env = [x,y,w/2,h/2,frame,roiMeanSigma,particules]
env = [-42,-42,8,22,None,None,None]
cv.namedWindow('win', 0)
cv.setMouseCallback('win', focus, env)

prvs = None
nxt = None

while 1:
	ret, env[4] = cap.read()
	if not ret:
		break
	#env[4] = cv.resize(env[4], (winSize, winSize))
	if prvs is None:
		prvs = cv.cvtColor(env[4],cv.COLOR_BGR2GRAY)
	else:
		nxt = cv.cvtColor(env[4],cv.COLOR_BGR2GRAY)
	
	a,b,c,d = env[1]-env[3],env[1]+env[3],env[0]-env[2],env[0]+env[2]
	if env[5] is None:
		cv.rectangle(env[4], (c,a), (d,b), (0,255,0), 1)
	elif env[6] is None:
		cv.rectangle(env[4], (c,a), (d,b), (255,0,0), 1)
	else:
		moveP(prvs, nxt, env)
		gaussianWeight(env)
		env[6] = resample(env)
		X = 0
		Y = 0
		for p in env[6]:
			x,y,w = p.ravel()
			X += x
			Y += y
			cv.circle(env[4],(int(x),int(y)),1,(255,0,0),-1)
		env[0], env[1] = int(X)/len(env[6]), int(Y)/len(env[6])
		a,b,c,d = env[1]-env[3],env[1]+env[3],env[0]-env[2],env[0]+env[2]
		cv.rectangle(env[4], (c,a), (d,b), (255,0,0), 1)
	
	cv.imshow('win', env[4])
	k = cv.waitKey(30) & 0xFF
	if not loop(k, env):
		break
	prvs = nxt if nxt is not None else prvs

#cap.release()
cv.destroyAllWindows()

pas = 8
coef = 4

def printEnvInfos(env):
	print("(x,y) = ({},{})".format(env[0], env[1]))
	print("(w/2,h/2) = ({},{})".format(env[2], env[3]))
	nbPix = (env[2] * 2 - 1)*(env[3] * 2 - 1)
	print("nbPixel in hist : {}".format(nbPix))
	if env[5]:
		print("<----ENV5----->")
		for i in range(2):
			print("	CANAL {}".format(i))
			print("		----> mu = {}".format(env[5][i][0]))
			print("		----> sigma = {}".format(env[5][i][1]))
		if env[6] is not None:
			print("<----ENV6----->")
			print(env[6])

def loop(k, env):
	global pas, coef
	if k == ord('q'):
		return 0
	else:
		if k == ord('s'):
			env[7] = 'display'
		if k == 57 and (pas == 8 or pas == coef * 8):
				pas = coef * 8 if pas == 8 else 8
		if k == 55 and (pas == 8 or pas == 8 / coef):
				pas = 8 / coef if pas == 8 else 8
		if k == 56:
			env[3] += pas
		if k == 50 and env[3] > pas:
			env[3] -= pas
		if k == 52 and env[2] > pas:
			env[2] -= pas
		if k == 54 and env[2]:
			env[2] += pas
		if k == ord('p'):
			printEnvInfos(env)
		if k == 103:
			env[5] = None
			env[6] = None
		if k == 114:
			env[6] = None
		return 1

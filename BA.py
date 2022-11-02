import numpy as np 
from random import randint


"""
	Levenberg-Matquardt Algorithm: argmin(delta) || r - J * delta ||^2 + lambda*|| delta ||^2
	Find 'delta': ( J.T*J + lambda * Id ) * delta = J.T * r
	lambda: Damping factor
	Dimensions:
		R: C -> 3x3 
		T: C -> 3x1 
		U: N -> 3x1 

		J: 2K x ( 6C+3N )
		r: 2K x 1
		delta: ( 6C+3N ) x 1

		J.T*J*delta: ( 6C+3N ) x ( 6C+3N )
		J.T*r: ( 6C+3N ) x 1

		K = sum(c=1 -> C) sum( n = 1 -> Nc )
		Nc: number of the points in the c-th frame.
"""




def Compute_Jacobian( R, T, U, P, K, j, i):
	"""
		j : index of the camera ( 1 <= j <= C )
		i : index of the 3D points ( 1 <= i <= N )
	"""
	Jp = np.zeros( (2,3) )
	xij, yij, zij = U[i][0], U[i][1], U[i][2]
	Jp[0,0] = 1/zij
	Jp[1,1] = 1/zij
	Jp[0,2] = -xij/zij**2
	Jp[1,2] = -yij/zij**2

	J = np.zeros( ( 3, 3*( len(R) + len(T) + len(U)))) 

	Gx = np.array( [[0, 0, 0], [0, 0, -1], [0, 1, 0]] )
	Gy = np.array( [[0, 0, 1], [0, 0, 0], [-1, 0, 0]] )
	Gz = np.array( [[0, -1, 0], [1, 0, 0], [0, 0, 0]] )

	for k in range( len(R)):
		if k == j:
			J[ :, 3*k:3*(k+1)] = np.array([ Gx @ U[i], Gy @ U[i], Gz @ U[i] ])

	for k in range( len(R), len(R)+len(T)):
		if k - len(R) == j:
			J[ :, 3*k:3*(k+1)] = -R[j].T

	for k in range( len(R)+len(T), len(R)+len(T)+len(U)):
		if k-len(R)-len(T) == i:
			J[ :, 3*k:3*(k+1)] = R[j].T

	return K[:2, :2] @ Jp @ J        # size: 2 x (3 * (2*C + N))


def Compute_Residual( R, T, U, P, K, j, i):
	"""
		j : index of the camera ( 1 <= j <= C )
		i : index of the 3D points ( 1 <= i <= N )
	"""

	point = P[i][j]
	projc = K @ R[j].T @ (U[i] - T[j])
	r = np.array( [ point[0] - projc[0]/projc[2], point[1] - projc[1]/projc[2]])

	return r


def Solve( J, r, lmbda):
	Id = np.diag( [1 for _ in range(J.shape[1])] )
	A = J.T @ J + lmbda * Id
	b = J.T @ r
	delta = np.linalg.solve( A, b) 

	return delta
	

def Update( R, T, U, delta):
	# Update R, T, U:
	Gx = np.array( [[0, 0, 0], [0, 0, -1], [0, 1, 0]] )
	Gy = np.array( [[0, 0, 1], [0, 0, 0], [-1, 0, 0]] )
	Gz = np.array( [[0, -1, 0], [1, 0, 0], [0, 0, 0]] )

	for c in range(C):
		dR = np.array([ Gx @ delta[6*c:6*c+3], Gy @ delta[6*c:6*c+3] , Gz @ delta[6*c:6*c+3]  ])
		R[c] = R[c] @ np.exp( dR)           #   exp(M) ~ 1 + M + M^2 ...
		T[c] = T[c] + delta[6*c+3:6*c+6]
	for n in range(N):
		U[n] = U[n] + delta[6*C+3*n:6*C+3*n+3]


def BA( R, T, U, P, K):
	"""
		R: Dictionnary include C rotation matrices of size 3x3
		R = { 1:M1, 2:M2, ..., C:Mc }

		T: Dictionnary include C translation matrices of size 1x3
		T = { 1:T1, 2:T2, ..., C:Tc }
		
		U: List of N points of size 1x3
		U = { 1:U1, 2:U2, ..., N:Un }
		
		P: Dictionnary of Lists, Projections of a 3D point Uj in the C camera Frames  
		P = { 	1:{ 1:p11, 2:p12, ..., C:p1C },
				2:{ 1:p21, 2:p22, ..., C:p2C },
				....
				N:{ 1:pN1, 2:pN2, ..., C:pNC }
			}
		pji = ( x, y ) or None 

		K: Calibration Matrix of size 3x3
		K = [[fx, 0, cx]
			 [0, fy, cy]
		 	 [0,  0, 1]]
	"""

	N = len( U)
	C = len( R)  # assert len( R) == len( T)


	# iteration = 0
	# residual = 1
	# while (residual > 0.5) and (iteration < 25):
	J = []
	r = [] 
	for j in range(C):
		for i in range(N):
			if P[i][j] != None:
				Jji = Compute_Jacobian( R, T, U, P, K, j, i)
				rji = Compute_Residual( R, T, U, P, K, j, i)
				J.append( Jji[0])
				J.append( Jji[1])
				r.append( rji[0])
				r.append( rji[1])
	
	J = np.array( J)
	r = np.array( r)

	# find 'delta' that: argmin(delta) ||rij - Jij * delta||^2 + lambda * ||delta||^2 
	lmbda = 0.01
	delta = Solve( J, r, lmbda )  # size (2*C+N) x 3 

	# Update R, T, U:
	Update( R, T, U, delta)

	# # Norm of the residual:
	# residual =  np.linalg.norm( r)
	# iteration += 1






####################################################################################################################
# Define Parameters: (Just for test)
####################################################################################################################
C = 2
N = 4


Gx = np.array( [[0, 0, 0], [0, 0, -1], [0, 1, 0]] )
Gy = np.array( [[0, 0, 1], [0, 0, 0], [-1, 0, 0]] )
Gz = np.array( [[0, -1, 0], [1, 0, 0], [0, 0, 0]] )

R = {}
for c in range(C):
	r = np.random.random( (3))
	R[c] = np.array([ Gx @ r, Gy @ r, Gz @ r])
T = { c:np.random.random( (3)) for c in range(C)}
U = { n:np.random.random( (3)) for n in range(N)}
P = { n:{ c:( n+c, n+c) for c in range(C)} for n in range(N) }
# Nn = [randint(0, N-1) for _ in range(N//4)]
# for n in Nn:
# 	Cc = [randint(0, C-1) for _ in range(C//4)]
# 	for c in Cc:
# 		P[n][c] = None # Some Points don't have a Match in Certain Camera Frames


H, W = 32, 32
K = np.array( [[100, 0, H/2], [0, 100, W/2], [0, 0, 1]] )

####################################################################################################################
# Apply Bundle Adjustment:
####################################################################################################################
print( R, "\n\n" )
BA( R, T, U, P, K)
print( "\n\n", R, "\n\n" )
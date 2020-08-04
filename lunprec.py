import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import optimize


# Input parameters
G = 6.67384 * 10**(-11)
M = 5.9723 * 10**(24)
m = 7.346 * 10**(22)
R = 1.4960 * 10**(11)
Omega = 1.9909866 * 10**(-7)
a = 3.844 * 10**(8)
ecc = 0.0549
epsilon = 8.9797 * 10**(-2)
va = 970
tsim = 1000000000
N = 100000


# Autocorrelation function
def autocorr(x):
    result = np.correlate(x, x, mode = 'full')
    return result[result.size//2:]


# Peak Finding function
def findpeaksind (arr):
	ret = []
	for i in range(1, len(arr) - 1):
		if (arr[i-1] <= arr[i]) and (arr[i+1] <= arr[i]):
			ret.append(i)
	return ret


# Arccos that returns angle mod 2*pi
def arccos2pi (x):
	ret = [np.arccos(x[0]),]
	for i in range(1, len(x)):
		if np.arccos(x[i]) >= np.arccos(x[i-1]):
			ret.append(np.arccos(x[i]))
		elif np.arccos(x[i]) < np.arccos(x[i-1]):
			ret.append(2*np.pi - np.arccos(x[i]))
	return np.array(ret)


# Differential equation
def dr_dt (r, t):
	A = -(Omega**2)*(R**3)*((r[0]**2 + r[2]**2 + r[4]**2)**(-3/2))
	B = -G*M*(((r[0] - R*np.cos(Omega*t))**2 + (r[2] - R*np.sin(Omega*t))**2 + r[4]**2)**(-3/2))
	return [r[1], A*r[0] + B*(r[0] - R*np.cos(Omega*t)), r[3], A*r[2] + B*(r[2] - R*np.sin(Omega*t)), r[5], A*r[4] + B*r[4]]


# Energy function
def energy (r, t):
	A = -(Omega**2)*(R**3)*((r[0]**2 + r[2]**2 + r[4]**2)**(-1/2))
	B = -G*M*(((r[0] - R*np.cos(Omega*t))**2 + (r[2] - R*np.sin(Omega*t))**2 + r[4]**2)**(-1/2))
	return (m/2)*(r[1]**2 + r[3]**2 + r[5]**2) + A*m + B*m - (1/2)*(M)*(R*Omega)**2


# Time and initial conditions
ts = np.linspace(0, tsim, N)
r0 = [R + a*(1 + ecc)*np.cos(epsilon), 0, 0, R*Omega + va, a*(1 + ecc)*np.sin(epsilon),0]


# Solving differential equation
rs = odeint(dr_dt, r0, ts)


# Energy conservation check
relpercE = (energy(np.transpose(rs), ts)/(energy(r0, 0)) - 1)*100


# Distance between Earth and Moon
distx = rs[:,0] - R*np.cos(Omega*ts)
disty = rs[:,2] - R*np.sin(Omega*ts)
distz = rs[:,4]
dist = ((distx)**2 + (disty)**2 + (distz)**2)**(1/2)


# Normal vector's components
ndotx = ((rs[:,2] - R*np.sin(Omega*ts))*rs[:,5] - rs[:,4]*(rs[:,3] - R*Omega*np.cos(Omega*ts)))/((((rs[:,2] - R*np.sin(Omega*ts))*rs[:,5] - rs[:,4]*(rs[:,3] - R*Omega*np.cos(Omega*ts)))**2 + (rs[:,4]*(rs[:,1] + R*Omega*np.sin(Omega*ts)) - (rs[:,0] - R*np.cos(Omega*ts))*rs[:,5])**2 + ((rs[:,0] - R*np.cos(Omega*ts))*(rs[:,3] - R*Omega*np.cos(Omega*ts)) - (rs[:,2] - R*np.sin(Omega*ts))*(rs[:,1] + R*Omega*np.sin(Omega*ts)))**2)**(1/2))
ndoty = (rs[:,4]*(rs[:,1] + R*Omega*np.sin(Omega*ts)) - (rs[:,0] - R*np.cos(Omega*ts))*rs[:,5])/((((rs[:,2] - R*np.sin(Omega*ts))*rs[:,5] - rs[:,4]*(rs[:,3] - R*Omega*np.cos(Omega*ts)))**2 + (rs[:,4]*(rs[:,1] + R*Omega*np.sin(Omega*ts)) - (rs[:,0] - R*np.cos(Omega*ts))*rs[:,5])**2 + ((rs[:,0] - R*np.cos(Omega*ts))*(rs[:,3] - R*Omega*np.cos(Omega*ts)) - (rs[:,2] - R*np.sin(Omega*ts))*(rs[:,1] + R*Omega*np.sin(Omega*ts)))**2)**(1/2))
ndotz = ((rs[:,0] - R*np.cos(Omega*ts))*(rs[:,3] - R*Omega*np.cos(Omega*ts)) - (rs[:,2] - R*np.sin(Omega*ts))*(rs[:,1] + R*Omega*np.sin(Omega*ts)))/((((rs[:,2] - R*np.sin(Omega*ts))*rs[:,5] - rs[:,4]*(rs[:,3] - R*Omega*np.cos(Omega*ts)))**2 + (rs[:,4]*(rs[:,1] + R*Omega*np.sin(Omega*ts)) - (rs[:,0] - R*np.cos(Omega*ts))*rs[:,5])**2 + ((rs[:,0] - R*np.cos(Omega*ts))*(rs[:,3] - R*Omega*np.cos(Omega*ts)) - (rs[:,2] - R*np.sin(Omega*ts))*(rs[:,1] + R*Omega*np.sin(Omega*ts)))**2)**(1/2))


# Remove DC component for autocorrelation
ndotx1 = ndotx - np.mean(ndotx)
ndoty1 = ndoty - np.mean(ndoty)
ndotz1 = ndotz - np.mean(ndotz)


# Calculating autocorrelation
atcx = autocorr(ndotx1)
atcy = autocorr(ndoty1)
atcz = autocorr(ndotz1)


# Find Peaks in autocorrelation
idx = findpeaksind(atcx)
idy = findpeaksind(atcy)
idz = findpeaksind(atcz)


# Period esstimates from different components
Tnodalx = (ts[int(idx[1])] - ts[int(idx[0])])/(1 * 31556926)
Tnodaly = (ts[int(idx[1])] - ts[int(idx[0])])/(1 * 31556926)
Tnodalz = (ts[int(idx[1])] - ts[int(idx[0])])/(1 * 31556926)


# Dot product of dist and xproj to measure angle
xprojx = 1 - (ndotx)**2
xprojy = -1 * ndotx*ndoty
xprojz = -1 * ndotx*ndotz

costheta = (distx*xprojx + disty*xprojy + distz*xprojz)/(dist)


# Fix numerical errors at ends of domain of arccos
for i in range(len(costheta)):
	if (costheta[i] > 1):
		costheta[i] = 1
	elif (costheta[i] < -1):
		costheta[i] = -1


# Find angle
theta = arccos2pi(costheta)


# Find apoapsides, their angles and their times
apoind = findpeaksind(dist)
apotheta = np.array([theta[i] for i in apoind])
apots = np.array([ts[i] for i in apoind])


# Find when angle jumps between 0 and 2*pi
index = []
for i in range(len(apotheta) - 1):
	if (np.abs(apotheta[i] - apotheta[i+1]) > 1.7*np.pi):
		index.append(i)


# Period calculation
Tapsidal = (apots[int(index[1])] - apots[int(index[0])])/(1 * 31556926)


# Normal vector rotates around z-axis
print('The axis of rotation of the normal vector has components:\n[{}, {}, {}]\n'.format(np.mean(ndotx), np.mean(ndoty), np.mean(ndotz)))


# Nodal period
print('The esstimates for the period of nodal precession from the differernt axes are:')
print('x-axis: {} years'.format(Tnodalx))
print('y-axis: {} years'.format(Tnodaly))
print('z-axis: {} years\n'.format(Tnodalz))


# Apsidal period
print('The esstimate for the period of apsidal precession is:\n{} years'.format(Tapsidal))


# Plotting

# Ploting components of normal vector
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif', size = 13)
plt.rc('legend', fontsize = 10, handlelength = 2)
plt.minorticks_on()

plt.plot(ts/31557600, ndotx, linestyle = '--', color = '#ff4c4c', label = '$x$-component')
plt.plot(ts/31557600, ndoty, linestyle = '-.', color = '#3e39a3', label = '$y$-component')
plt.plot(ts/31557600, ndotz1, linestyle = '-', color = '#47a238', label = '$z$-component')
plt.xlabel(r'Time $t \, [\mathrm{years}]$')
plt.ylabel(r'Components of $\mathrm{\mathbf{n}} - \langle \mathrm{\mathbf{n}} \rangle$')
plt.legend()

plt.tight_layout()
plt.savefig('test.pdf', format = 'pdf', dpi = 1200)
plt.show()
plt.close()


# Plotting angles of apoapsides vs time
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif', size = 13)
plt.minorticks_on()

plt.plot(apots/31557600, apotheta, linestyle = '-', color = '#ff4c4c')
plt.xlabel(r'Time of apoapsis $t_a \, [\mathrm{years}]$')
plt.ylabel(r'Angle of apoapsis $\theta_a \, (\mathrm{mod} \; 2 \pi)$')

plt.tight_layout()
plt.savefig('test.pdf', format = 'pdf', dpi = 1200)
plt.show()
plt.close()

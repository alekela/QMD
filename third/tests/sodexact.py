import math
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

from simpleSPH.data import field, Storage

def shockResolve(P, mach_L, gamma, PRL, CRL):
	return (1.0 + mach_L*(gamma-1.0)/2.0-(gamma-1.0)*CRL*(P-1.0)/math.sqrt(2.0*gamma*(gamma-1.0+(gamma+1.0)*P)))**(2.0*gamma/(gamma-1.0))/P-PRL

def solution(gamma, dens_L, v_L, pres_L, dens_R, v_R, pres_R, time, nPoints):
	dtype = np.dtype([
		('x', 'f8'), 
		('velocity', 'f8'), 
		('density', 'f8'), 
		('pressure', 'f8'), 
		('energy', 'f8'), 
		('soundSpeed', 'f8'),
		('mach', 'f8')
	])

	out = np.zeros(nPoints, dtype=dtype)
	alpha=(gamma+1.0)/(gamma-1.0)
	PRL = pres_R/pres_L
	c_L = math.sqrt(gamma*pres_L/dens_L)
	c_R = math.sqrt(gamma*pres_R/dens_R)
	CRL = c_R/c_L
	mach_L = (v_L - v_R)/c_L
	pres_MR = pres_R*optimize.root(shockResolve, x0=3.0, args=(mach_L, gamma, PRL, CRL)).x
	dens_MR = dens_R*(1.0+alpha*pres_MR/pres_R)/(alpha+pres_MR/pres_R)
	dens_ML = dens_L*(pres_MR/pres_L)**(1.0/gamma)
	v_ML = v_L-v_R+(2.0/(gamma-1.0))*c_L*(1.0-(pres_MR/pres_L)**((gamma-1.0)/(2.0*gamma)))
	c_ML = math.sqrt(gamma*pres_MR/dens_ML)
	spos = 0.5+time*c_R*math.sqrt((gamma-1.0)/(2.0*gamma)+(gamma+1.0)/(2.0*gamma)*pres_MR/pres_R)+time*v_R

	x0 = 0.5
	conpos = x0 + v_ML*time+time*v_R    # Position of contact discontinuity
	pos1   = x0 + (v_L-c_L)*time        # Start of expansion fan
	pos2   = x0 + (v_ML+v_R-c_ML)*time  # End of expansion fan

	out["x"] = np.linspace(0.0, 1.0, nPoints)
	ipos1    = np.nonzero(np.greater_equal(out["x"], 0.0)*np.less_equal(out["x"], pos1))
	ipos2    = np.nonzero(np.greater(out["x"], pos1)* np.less_equal(out["x"], pos2))
	iconpos  = np.nonzero(np.greater(out["x"], pos2)* np.less_equal(out["x"], conpos))
	ispos    = np.nonzero(np.greater(out["x"], conpos)* np.less_equal(out["x"], spos))
	ifinal   = np.nonzero(np.greater(out["x"], spos)* np.less_equal(out["x"], 1.0))

	out["pressure"][ipos1] = pres_L
	out["density"][ipos1]  = dens_L
	out["velocity"][ipos1] = v_L

	out["pressure"][ipos2] = pres_L*(1.0+(pos1-out["x"][ipos2])/(c_L*alpha*time))**(2.0*gamma/(gamma-1.0))
	out["density"][ipos2]  = dens_L*(1.0+(pos1-out["x"][ipos2])/(c_L*alpha*time))**(2.0/(gamma-1.0))
	out["velocity"][ipos2] = v_L + (2.0/(gamma+1.0))*(out["x"][ipos2]-pos1)/time

	out["pressure"][iconpos] = pres_MR
	out["density"][iconpos]  = dens_ML
	out["velocity"][iconpos] = v_ML+v_R

	out["pressure"][ispos] = pres_MR
	out["density"][ispos]  = dens_MR
	out["velocity"][ispos] = v_ML+v_R

	out["pressure"][ifinal] = pres_R
	out["density"][ifinal]  = dens_R
	out["velocity"][ifinal] = v_R

	out["energy"] = out["pressure"]/((gamma - 1.0)*out["density"])
	out["soundSpeed"] = np.sqrt(gamma*out["pressure"]/out["density"])
	out["mach"] = out["velocity"]/out["soundSpeed"]

	storage = Storage(
		fields=[
			field.coords,
			field.density,
			field.velocity,
			field.pressure,
			field.energy
		], 
		size = nPoints
	)

	d = 1.0/nPoints
	for i in range(nPoints):
		storage[field.coords][i]   = -0.5 + 0.5*d + i*d
		storage[field.density][i]  = out["density"][i]
		storage[field.velocity][i] = out["velocity"][i]
		storage[field.pressure][i] = out["pressure"][i]
		storage[field.energy][i]   = out["energy"][i]

	return storage

if (__name__ == "__main__"):
	data = solution(1.4, 1.0, 0.75, 1.0, 0.125, 0.0, 0.1, 0.2, 300)
	grid = gs.GridSpec(2, 2)

	density  = plt.subplot(grid[0,0])
	velocity = plt.subplot(grid[0,1])
	pressure = plt.subplot(grid[1,0])
	energy   = plt.subplot(grid[1,1])

	density.plot(data["x"] - 0.2,  data["density"])
	density.set_xlim(0.0, 1.0)
	velocity.plot(data["x"] - 0.2, data["velocity"])
	velocity.set_xlim(0.0, 1.0)
	pressure.plot(data["x"] - 0.2, data["pressure"])
	pressure.set_xlim(0.0, 1.0)
	energy.plot(data["x"] - 0.2,   data["energy"])
	energy.set_xlim(0.0, 1.0)
	# energy.set_ylim(0.0, 1.2*np.amax(data["energy"]))

	plt.show()
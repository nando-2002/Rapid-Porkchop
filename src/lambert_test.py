import numpy as np
from utils.uplanet import uplanet
from utils.utils import date2jd, date2mjd2000, jd2date, kep2car, car2kep
from utils.astroConstants import astroConstants
from utils.lambert import lam_solve
from utils.plot_orbit import plot_two_orbits, plot_two_orbits_from_states, plot_three_orbits_from_states


'''
r1 = np.array([-1.09993019e+08, 9.83368937e+07,  0.00000000e+00]) #km
r2 = np.array([-2.44947265e+08, -2.24360841e+07,  5.55318368e+06]) #km

TOF =  17596714.00001049 #s
mu = astroConstants(4) #km^3/s^2
v1, v2 = lam_solve(r1, r2, TOF, mu)

print(f"V1 : ", v1)
print(f"V2 : ", v2)

h, a, e, Om, i, om, theta = car2kep(r1, v1, mu)

print(f"Semi major axis : ", a, "km")
'''

# heliocentric so, 
mu = astroConstants(4)

# Let us take Jun 7, 2003 at 22:27:34
departure = np.array([2003, 6, 7, 22, 27, 34]) 
departure_mjd = date2mjd2000(departure)

# the planet Earth 
a, e, i, Om, om, theta = uplanet(departure_mjd, 3)
h = np.sqrt( mu * a * ( 1 - e**2 ) )

orb1 = (a, e, i, Om, om)


# Let us take Jun 7, 2003 at 22:27:34
arrival = np.array([2003, 12, 28, 14, 26, 8]) 
arrival_mjd = date2mjd2000(arrival)

# the planet Earth 
aa, ea, ia, Oma, oma, thetaa = uplanet(arrival_mjd, 4)
ha = np.sqrt( mu * aa * ( 1 - ea**2 ) )

orb2 = (aa, ea, ia, Oma, oma)
#plot_two_orbits(orb1, orb2, labels = ("Earth", "Mars"))

# completed the initial plotting and whatnot

# now to actually do some computation 

# computing cartesian coordinates for the planets 

#Earth
re, ve = kep2car(h, e, i, Om, om, theta, mu)
re = np.asarray(re).flatten()
ve = np.asarray(ve).flatten()

#Mars
rm, vm = kep2car(ha, ea, ia, Oma, oma, thetaa, mu)
rm = np.asarray(rm).flatten()
vm = np.asarray(vm).flatten()

#plot_two_orbits_from_states(re, ve, rm, vm, mu, labels = ("Earth", "Mars") )


# now we proceed to do the lambert problem

TOF = (arrival_mjd - departure_mjd)*24*60*60 # days -> seconds
rt, vt = lam_solve(re, rm, TOF, mu, 10_000)

plot_two_orbits_from_states(re, ve, rt, vt, mu)
print(rt)
print(vt)
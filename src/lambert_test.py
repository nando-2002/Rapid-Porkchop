import numpy as np
from utils.uplanet import uplanet
from utils.utils import date2jd, date2mjd2000, jd2date, kep2car, car2kep
from utils.astroConstants import astroConstants
from utils.lambert import lam_solve
from utils.plot_orbit import plot_two_orbits, plot_two_orbits_from_states, plot_three_orbits_from_states


# heliocentric so, 
mu = astroConstants(4)

# Let us take Jun 7, 2003 at 22:27:34
departure = np.array([2003, 6, 7, 22, 27, 34]) 
departure_mjd = date2mjd2000(departure)

# the planet Earth 
a, e, i, Om, om, theta = uplanet(departure_mjd, 3)
h = np.sqrt( mu * a * ( 1 - e**2 ) )

orb1 = (a, e, i, Om, om)


# Time of arrival at Mars : Dec 28, 2003 at 14:26:08 (known from the ppt)
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
vt, vt2 = lam_solve(re, rm, TOF, mu, 10_000)

# something is wrong here 
H, A, E, OMM, I, OM, THETA =  car2kep(vt, vt2, mu)

orb3 = (A, E, I, OMM, OM)

#plot_two_orbits(orb1, orb3, labels = ("Earth", "Transfer Orbit") )
print(vt)
print(vt2)
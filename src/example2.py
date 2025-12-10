# Uplanet.py test with known data
# ----------------------------------------------------------------------
import numpy as np
from utils.uplanet import uplanet
from utils.utils import date2jd, date2mjd2000, jd2date, kep2car
from utils.astroConstants import astroConstants
from utils.lambert import lam_solve

departure = np.array([2003, 6, 7, 22, 27, 34]) 
departure_mjd = date2mjd2000(departure)

r1_kep = uplanet(departure_mjd, 3) # departure from earth 
SunMu = astroConstants(4)

h_earth = np.sqrt( SunMu * r1_kep[0] * ( 1 - r1_kep[1]**2) )

r1_car, v1_car = kep2car(h_earth, r1_kep[1], r1_kep[2], r1_kep[3], r1_kep[4], r1_kep[5], SunMu)

print("----------------------------------------------------------------------")
print("Earth at Departure")
print(f"Position Vector ", r1_car)
print(f"Velocity Vector", v1_car)
print(f"Velocity Mag", np.linalg.norm(v1_car))


arrival = np.array([2003, 12, 28, 14, 26, 8])
arrival_mjd = date2mjd2000(arrival)

r2_kep = uplanet(arrival_mjd, 4) # departure from earth 

h_mars = np.sqrt( SunMu * r2_kep[0] * ( 1 - r2_kep[1]**2) )

r2_car, v2_car = kep2car(h_mars, r2_kep[1], r2_kep[2], r2_kep[3], r2_kep[4], r2_kep[5], SunMu)

print("----------------------------------------------------------------------")
print("Mars at Arrival")
print(f"Position Vector", r2_car)
print(f"Velocity Vector", v2_car)
print(f"Velocity Mag", np.linalg.norm(v2_car))

TOF = (arrival_mjd - departure_mjd) * 24 * 60 * 60
r1_car = np.asarray(r1_car).flatten()
r2_car = np.asarray(r2_car).flatten()

print(r2_car)

v1, v2 = lam_solve(r1_car, r2_car, TOF, SunMu)

delv1 = v1 - v1_car
delv2 = v2 - v2_car

tot_del_v = np.linalg.norm(delv1) + np.linalg.norm(delv2)

print("----------------------------------------------------------------------")
print(f"Total Delta V \n", tot_del_v, "km/s")
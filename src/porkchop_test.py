import numpy as np
from utils.uplanet import uplanet
from utils.utils import date2jd, date2mjd2000, jd2date, kep2car, car2kep
from utils.astroConstants import astroConstants
from utils.lambert import lam_solve
from utils.plot_orbit import plot_two_orbits, plot_two_orbits_from_states, plot_three_orbits_from_states


# heliocentric so, 
mu = astroConstants(4)

# departure date range : 2 April 2003 to 1 August 2003
departure_start = np.array([2003, 4, 2, 0, 0, 0]) 
departure_end = np.array([2003, 8, 1, 0, 0, 0])

#arrival date range : 1 Sept 2003 to 1 March 2004
arrival_start = np.array([2003, 9, 1, 0, 0, 0]) 
arrival_end = np.array([2004, 3, 1, 0, 0, 0])

#convert to mjd2000
departure_start_mjd = date2mjd2000(departure_start)
departure_end_mjd = date2mjd2000(departure_end)
arrival_start_mjd = date2mjd2000(arrival_start)
arrival_end_mjd = date2mjd2000(arrival_end)

#create a range of departure and arrival dates
npoints = 25 # points per axis
dep_range = np.linspace(departure_start_mjd, departure_end_mjd, npoints)
arr_range = np.linspace(arrival_start_mjd, arrival_end_mjd, npoints)

delta_v_solutions = np.zeros((npoints, npoints)) # to store delta-v values
for outcount in range(npoints):
    for incount in range(npoints):
        dep_mjd = dep_range[outcount]
        arr_mjd = arr_range[incount]
        
        if arr_mjd <= dep_mjd:
            continue # arrival must be after departure
        
        # get planet states at departure and arrival
        # Earth at departure
        a, e, i, Om, om, theta = uplanet(dep_mjd, 3)
        h = np.sqrt( mu * a * ( 1 - e**2 ) )
        re, ve = kep2car(h, e, i, Om, om, theta, mu)
        re = np.asarray(re).flatten()
        ve = np.asarray(ve).flatten()
        
        # Mars at arrival
        aa, ea, ia, Oma, oma, thetaa = uplanet(arr_mjd, 4)
        ha = np.sqrt( mu * aa * ( 1 - ea**2 ) )
        rm, vm = kep2car(ha, ea, ia, Oma, oma, thetaa, mu)
        rm = np.asarray(rm).flatten()
        vm = np.asarray(vm).flatten()
        
        # time of flight
        tof = (arr_mjd - dep_mjd) * 86400  # convert days to seconds
        
        # solve Lambert's problem
        try:
            v1_sol, v2_sol = lam_solve(re, rm, tof, mu)
            # Here you can store or process the solutions as needed
        except Exception as e:
            print(f"Lambert solver failed for dep {dep_mjd}, arr {arr_mjd}: {e}")
            continue
        
        # compute total delta v
        delta_v_solutions[outcount,incount] = np.linalg.norm(v1_sol - ve) + np.linalg.norm(vm - v2_sol)

# find the lowest delta-v solution
del_v_lowest = np.min(delta_v_solutions[np.nonzero(delta_v_solutions)])
print(del_v_lowest)
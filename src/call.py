from lambert import lam_solve
import numpy as np

r1 = np.array([-21800, 37900, 0]) #km
r2 = np.array([27300, 27700, 0]) #km

delT = 15*60*60 + 6*60 + 40 #s
mu = 3.986*10**5  #km^3/s^2
v1, v2 = lam_solve(r1, r2, delT, mu)

print(f"V1 : ", v1)
print(f"V2 : ", v2)

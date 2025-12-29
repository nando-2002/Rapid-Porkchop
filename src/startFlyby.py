# startFlyby.py (same location as startPorkchop.py)
from gpu.call_flyby import call_flyby
from utils.plotflyby import save_flyby_plot, save_flyby_isosurface

soln, dep, fly, arr, r, v = call_flyby(50, 3, 4, 5)  # Earth->Mars flyby->Jupiter
results = (soln, dep, fly, arr)

save_flyby_plot(results, "Earth_MarsFlyby_Jupiter.png")

save_flyby_isosurface(results, filename="Earth_MarsFlyby_Jupiter_isosurface.png")

# Or specify an explicit isovalue (ΔV in km/s) and threshold
save_flyby_isosurface(results,
                      filename="Earth_MarsFlyby_Jupiter_isosurface2.png",
                      level=15.0,             # isosurface at ΔV = 15 km/s
                      min_dv_threshold=30)    # mask values > 30 km/s as invalid
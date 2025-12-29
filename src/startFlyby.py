# startFlyby.py (same location as startPorkchop.py)
from gpu.call_flyby import call_flyby
from utils.plotflyby import save_flyby_plot

soln, dep, fly, arr, r, v = call_flyby(50, 3, 4, 5)  # Earth->Mars flyby->Jupiter
results = (soln, dep, fly, arr)
save_flyby_plot(results, "Earth_MarsFlyby_Jupiter.png")

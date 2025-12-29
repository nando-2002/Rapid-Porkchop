# startFlyby.py (same location as startPorkchop.py)
from gpu.call_flyby import call_flyby
from utils.plotflyby import save_flyby_plot, get_optimal_dates, save_flyby_density_plot

soln, dep, fly, arr, r, v = call_flyby(50, 3, 4, 11)  # Earth->Mars flyby->Jupiter
results = (soln, dep, fly, arr)

# Print optimal dates to console
dep_str, fly_str, arr_str, min_val = get_optimal_dates(results)
print(f"Optimal dates (min ΔV={min_val:.2f} km/s):\n  Departure: {dep_str}\n  Flyby:     {fly_str}\n  Arrival:   {arr_str}")

save_flyby_plot(results, "Earth_MarsFlyby_257323.png")
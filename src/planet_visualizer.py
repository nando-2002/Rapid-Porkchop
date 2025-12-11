import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import os

# Planetary diameters in relative units (normalized to Earth = 1)
PLANET_DIAMETERS = {
    "sun": 109.0,
    "mercury": 0.383,
    "venus": 0.949,
    "earth": 1.0,
    "mars": 0.532,
    "jupiter": 11.21,
    "saturn": 9.45,
    "uranus": 4.01,
    "neptune": 3.88
}

def _find_texture_file(img_dir, planet_key):
    # Busca con varias extensiones comunes
    exts = [".jpg", ".jpeg", ".png", ".tif", ".bmp"]
    for ext in exts:
        candidate = os.path.join(img_dir, f"{planet_key}{ext}")
        if os.path.exists(candidate):
            return candidate
    return None

def _sample_texture_on_sphere(img, theta, phi):
    """
    img: numpy array HxWxC (uint8 or float)
    theta: (M,N) array in [0, 2*pi]
    phi: (M,N) array in [0, pi]
    devuelve: facecolors (M,N,4) float in [0,1]
    """
    H, W = img.shape[:2]
    # UV coords: u along theta, v along phi (flip v to match typical images)
    u = theta / (2.0 * np.pi)  # 0..1
    v = 1.0 - (phi / np.pi)    # 0..1 (flip vertical so poles map correctly)

    # Map to pixel indices
    x = (u * (W - 1)).astype(int)
    y = (v * (H - 1)).astype(int)

    sampled = img[y, x]
    # Ensure RGBA
    if sampled.shape[2] == 3:
        alpha = np.ones(sampled.shape[:2] + (1,), dtype=sampled.dtype)
        sampled = np.concatenate([sampled, alpha], axis=2)
    # Normalize to float 0..1
    sampled = sampled.astype(np.float32) / 255.0
    return sampled

def create_planet_sphere(planet_name, scale, img_dir="img", resolution=300):
    """
    Creates a textured sphere for a planet and adds it to a matplotlib figure.
    
    Parameters:
    - planet_name (str): Name of the planet (e.g., 'Earth', 'Mars', 'Sun')
    - scale (float): Scale factor multiplied by planet's actual diameter
    - img_dir (str): Directory containing planet texture images (relative to this file by default)
    - resolution (int): sphere resolution (higher = smoother)
    
    Returns:
    - fig: matplotlib figure object
    - ax: 3D axis object
    """
    
    # Get planet diameter and calculate radius
    planet_key = planet_name.lower()
    diameter = PLANET_DIAMETERS.get(planet_key, 1.0)
    radius = (diameter / 2.0) * scale
    
    # Resolve img_dir relative to this file if not absolute
    if not os.path.isabs(img_dir):
        base_dir = os.path.dirname(__file__)
        img_dir = os.path.join(base_dir, img_dir)
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create sphere mesh (theta: azimuth, phi: polar)
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution//2)
    theta, phi = np.meshgrid(u, v)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    
    # Try to find texture file
    texture_path = _find_texture_file(img_dir, planet_key)
    
    if texture_path and os.path.exists(texture_path):
        img = Image.open(texture_path).convert("RGBA")
        img_np = np.array(img)
        facecolors = _sample_texture_on_sphere(img_np, theta, phi)
        ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=facecolors,
                        linewidth=0, antialiased=False, shade=False)
    else:
        # Default coloring if file not found: simple shaded surface
        from matplotlib import cm
        ls = plt.cm.viridis
        # use z for coloring
        norm = (z - z.min()) / (z.max() - z.min() + 1e-9)
        colors = cm.viridis(norm)
        ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=colors,
                        linewidth=0, antialiased=True, shade=True)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"{planet_name} (Diameter: {diameter:.2f}x, Scale: {scale}x)")
    ax.set_box_aspect([1,1,1])
    # Turn off grid for cleaner look
    ax.grid(False)
    # Adjust view
    ax.view_init(elev=20, azim=30)
    
    return fig, ax


def visualize_solar_system(scale_factor=1.0, img_dir="img"):
    """
    Visualizes all planets in the solar system with their textures.
    
    Parameters:
    - scale_factor (float): Scale factor multiplied by each planet's actual diameter
    - img_dir (str): Directory containing planet texture images
    """
    
    planets = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
    
    figures = {}
    for planet in planets:
        fig, ax = create_planet_sphere(planet, scale_factor, img_dir)
        figures[planet] = (fig, ax)
    
    plt.show()
    return figures


if __name__ == "__main__":
    # Example usage
    #visualize_solar_system(scale_factor=1.0, img_dir="img")

    create_planet_sphere("Earth", 1.0, img_dir="img")
    plt.show()

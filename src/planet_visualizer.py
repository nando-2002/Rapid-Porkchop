import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import os

def create_planet_sphere(planet_name, scale, img_dir="/img"):
    """
    Creates a textured sphere for a planet and adds it to a matplotlib figure.
    
    Parameters:
    - planet_name (str): Name of the planet (e.g., 'Earth', 'Mars', 'Sun')
    - scale (float): Scale factor for the sphere radius
    - img_dir (str): Directory containing planet texture images
    
    Returns:
    - fig: matplotlib figure object
    - ax: 3D axis object
    """
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create sphere mesh
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = scale * np.outer(np.cos(u), np.sin(v))
    y = scale * np.outer(np.sin(u), np.sin(v))
    z = scale * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Load texture
    texture_path = os.path.join(img_dir, f"{planet_name.lower()}.jpg")
    
    if os.path.exists(texture_path):
        texture = Image.open(texture_path)
        texture = np.array(texture)
        # Normalize texture to [0, 1]
        if texture.dtype == np.uint8:
            texture = texture / 255.0
        # Use red channel or convert if needed
        if len(texture.shape) == 3:
            texture = texture[:, :, 0]
    else:
        # Default texture if file not found
        texture = np.ones((100, 100))
    
    # Resize texture to match mesh
    from scipy.ndimage import zoom
    texture = zoom(texture, (x.shape[0] / texture.shape[0], x.shape[1] / texture.shape[1]))
    
    # Plot surface with texture
    ax.plot_surface(x, y, z, facecolors=plt.cm.gray(texture), shade=False)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"{planet_name} (Scale: {scale})")
    ax.set_box_aspect([1,1,1])
    
    return fig, ax


def visualize_solar_system(scale_factor=1.0, img_dir="/img"):
    """
    Visualizes all planets in the solar system with their textures.
    
    Parameters:
    - scale_factor (float): Scale factor for all planets
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
    visualize_solar_system(scale_factor=1.0, img_dir="/img")

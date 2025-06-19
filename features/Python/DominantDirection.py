import numpy as np
from scipy import signal

def compute_orientation_and_coherence(img):
    """
    Computes dominant orientation (degrees) and coherence (fraction) for a 2D image.
    Strictly replicates the Java method `computeSpline` from OrientationJ.
    
    Args:
        img: 2D numpy array (grayscale image)
        
    Returns:
        tuple: (orientation_degrees, coherence_fraction)
    """
    # Convert to float for gradient calculations
    img = img.astype(np.float32)
    h, w = img.shape
    
    # Handle small images (no inner region)
    if h < 3 or w < 3:
        return (0.0, 0.0)
    
    # Cubic spline gradient kernels
    k = np.array([1, -8, 0, 8, -1], dtype=np.float32) / 12.0
    kernel_x = k.reshape(1, 5)  # X-gradient kernel (1x5)
    kernel_y = k.reshape(5, 1)  # Y-gradient kernel (5x1)
    
    # Convolve with zero padding (matches Java behavior)
    dx = signal.convolve2d(img, kernel_x, mode='same', boundary='fill', fillvalue=0)
    dy = signal.convolve2d(img, kernel_y, mode='same', boundary='fill', fillvalue=0)
    
    # Remove 1-pixel border (equivalent to Java's inner region)
    dx_inner = dx[1:h-1, 1:w-1]
    dy_inner = dy[1:h-1, 1:w-1]
    
    # Compute structure tensor components (averaged)
    vxx = np.mean(dx_inner**2)
    vyy = np.mean(dy_inner**2)
    vxy = np.mean(dx_inner * dy_inner)
    
    # Calculate orientation (degrees)
    orientation_rad = 0.5 * np.arctan2(2 * vxy, vyy - vxx)
    orientation_deg = np.degrees(orientation_rad)
    
    # Calculate coherence (fraction, 0.0 if denominator too small)
    d = vyy - vxx
    numerator = np.sqrt(d**2 + 4 * vxy**2)
    denominator = vxx + vyy
    coherence = numerator / denominator if denominator > 1 else 0.0
    
    return (orientation_deg, coherence)

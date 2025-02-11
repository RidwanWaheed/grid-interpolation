"""
Test Data Generator for XYZ Grid Interpolation

This script generates various test cases for the xyz to grid interpolation program.
Each test case demonstrates different challenging scenarios.

Test cases include:
1. Regular grid-like points
2. Random scattered points
3. Clustered points with sparse areas
4. Collinear points
5. Points forming elongated triangles
6. Points with extreme z-value variations
7. Sparse boundary points
"""

import numpy as np
import os

def ensure_output_dir(dir_name="test_data"):
    """Create output directory if it doesn't exist"""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

def save_xyz_data(points, filename, dir_name="test_data"):
    """Save points to ASCII file"""
    filepath = os.path.join(dir_name, filename)
    np.savetxt(filepath, points, fmt='%.6f', delimiter=' ',
               header='X Y Z', comments='')
    print(f"Created test file: {filepath} with {len(points)} points")

def create_regular_grid_points(nx=10, ny=10, noise=0.1):
    """
    Case 1: Nearly regular grid points with small random offsets
    Tests basic interpolation functionality
    """
    x = np.linspace(0, 100, nx)
    y = np.linspace(0, 100, ny)
    xx, yy = np.meshgrid(x, y)
    
    # Add small random offsets
    xx += np.random.normal(0, noise, xx.shape)
    yy += np.random.normal(0, noise, yy.shape)
    
    # Create smooth z values (example: a hill)
    zz = np.exp(-((xx-50)**2 + (yy-50)**2) / 1000)
    
    points = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))
    save_xyz_data(points, "case1_regular_grid.txt")
    return points

def create_random_scattered_points(n_points=100):
    """
    Case 2: Random scattered points
    Tests interpolation with irregular point distribution
    """
    x = np.random.uniform(0, 100, n_points)
    y = np.random.uniform(0, 100, n_points)
    
    # Create z values (example: multiple hills)
    z = np.zeros(n_points)
    for cx, cy in [(25,25), (75,75), (25,75), (75,25)]:
        z += np.exp(-((x-cx)**2 + (y-cy)**2) / 500)
    
    points = np.column_stack((x, y, z))
    save_xyz_data(points, "case2_scattered_points.txt")
    return points

def create_clustered_points():
    """
    Case 3: Clustered points with sparse areas
    Tests handling of areas with insufficient data
    """
    # Create three clusters
    clusters = []
    centers = [(20,20), (60,60), (80,20)]
    
    for cx, cy in centers:
        n_points = np.random.randint(30, 50)
        cluster_x = np.random.normal(cx, 5, n_points)
        cluster_y = np.random.normal(cy, 5, n_points)
        cluster_z = np.exp(-((cluster_x-cx)**2 + (cluster_y-cy)**2) / 100)
        clusters.append(np.column_stack((cluster_x, cluster_y, cluster_z)))
    
    points = np.vstack(clusters)
    save_xyz_data(points, "case3_clustered_points.txt")
    return points

def create_collinear_points():
    """
    Case 4: Nearly collinear points
    Tests handling of potentially problematic triangulation
    """
    # Create main line
    t = np.linspace(0, 100, 50)
    x = t
    y = t + np.random.normal(0, 0.1, len(t))  # Nearly collinear
    z = np.sin(t/10)
    
    # Add a few off-line points to allow triangulation
    n_extra = 5
    extra_x = np.random.uniform(0, 100, n_extra)
    extra_y = np.random.uniform(0, 100, n_extra)
    extra_z = np.random.uniform(-1, 1, n_extra)
    
    points = np.vstack((
        np.column_stack((x, y, z)),
        np.column_stack((extra_x, extra_y, extra_z))
    ))
    save_xyz_data(points, "case4_collinear_points.txt")
    return points

def create_elongated_triangles():
    """
    Case 5: Points that will form elongated triangles
    Tests edge length checking
    """
    # Create sparse points in center
    x = np.linspace(0, 100, 5)
    y = np.linspace(0, 100, 5)
    xx, yy = np.meshgrid(x, y)
    
    # Add dense points at borders
    border_x = np.linspace(0, 100, 50)
    border_points = []
    
    for bx in border_x:
        border_points.append([bx, 0, np.sin(bx/10)])
        border_points.append([bx, 100, np.sin(bx/10)])
    
    for by in y:
        border_points.append([0, by, np.sin(by/10)])
        border_points.append([100, by, np.sin(by/10)])
    
    points = np.vstack((
        np.column_stack((xx.flatten(), yy.flatten(), 
                        np.zeros(xx.size))),
        np.array(border_points)
    ))
    save_xyz_data(points, "case5_elongated_triangles.txt")
    return points

def create_extreme_z_variations():
    """
    Case 6: Points with extreme z-value variations
    Tests interpolation with large value ranges
    """
    n_points = 100
    x = np.random.uniform(0, 100, n_points)
    y = np.random.uniform(0, 100, n_points)
    
    # Create extreme variations
    z = np.zeros(n_points)
    for i, (px, py) in enumerate(zip(x, y)):
        if 40 < px < 60 and 40 < py < 60:
            z[i] = 1000  # Extreme peak
        else:
            z[i] = np.random.uniform(-10, 10)
    
    points = np.column_stack((x, y, z))
    save_xyz_data(points, "case6_extreme_z_values.txt")
    return points

def create_sparse_boundary():
    """
    Case 7: Sparse points near boundary
    Tests edge behavior
    """
    # Dense interior points
    interior_x = np.random.uniform(20, 80, 80)
    interior_y = np.random.uniform(20, 80, 80)
    interior_z = np.exp(-((interior_x-50)**2 + (interior_y-50)**2) / 1000)
    
    # Sparse boundary points
    boundary_points = []
    for _ in range(20):
        # Random points near boundaries
        if np.random.random() < 0.5:
            x = np.random.uniform(0, 100)
            y = np.random.choice([np.random.uniform(0, 10), 
                                np.random.uniform(90, 100)])
        else:
            x = np.random.choice([np.random.uniform(0, 10), 
                                np.random.uniform(90, 100)])
            y = np.random.uniform(0, 100)
        z = np.random.uniform(-1, 1)
        boundary_points.append([x, y, z])
    
    points = np.vstack((
        np.column_stack((interior_x, interior_y, interior_z)),
        np.array(boundary_points)
    ))
    save_xyz_data(points, "case7_sparse_boundary.txt")
    return points

def create_all_test_cases():
    """Generate all test cases"""
    output_dir = ensure_output_dir()
    
    # Generate each test case
    create_regular_grid_points()
    create_random_scattered_points()
    create_clustered_points()
    create_collinear_points()
    create_elongated_triangles()
    create_extreme_z_variations()
    create_sparse_boundary()

if __name__ == "__main__":
    create_all_test_cases()

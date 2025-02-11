#!/usr/bin/env python3
"""
XYZ Point Data to Grid Interpolation

This program converts irregular point measurements (xyz data) into a regular grid 
while handling areas with insufficient data coverage. The program uses Delaunay 
triangulation for interpolation and flags areas far from measurements as NoData cells.

Basic workflow:
1. Read xyz point data from ASCII file
2. Create Delaunay triangulation of points
3. Create regular grid
4. Interpolate values within triangles
5. Flag areas far from measurements as NoData
6. Visualize results

Author: Ridwan
Date: January 2025
"""

import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import logging
import os

def read_xyz_data(filepath: str) -> Optional[np.ndarray]:
    """
    Read and validate ASCII xyz data file.
    
    Args:
        filepath: Path to input ASCII file containing xyz coordinates
        
    Returns:
        numpy.ndarray: Array of shape (n,3) containing xyz coordinates
        None: If file reading or validation fails
    """
    try:
        # Read data, skipping the header row
        data = np.loadtxt(filepath, skiprows=1)
        
        # Validate data shape
        if len(data.shape) != 2 or data.shape[1] != 3:
            logging.error(f"Expected shape (n,3), found shape {data.shape}")
            return None
            
        logging.info(f"Successfully read {len(data)} points")
        logging.info(f"X range: [{np.min(data[:,0]):.2f}, {np.max(data[:,0]):.2f}]")
        logging.info(f"Y range: [{np.min(data[:,1]):.2f}, {np.max(data[:,1]):.2f}]")
        logging.info(f"Z range: [{np.min(data[:,2]):.2f}, {np.max(data[:,2]):.2f}]")
        return data
        
    except Exception as e:
        logging.error(f"Failed to read input file: {str(e)}")
        return None

def setup_grid(points: np.ndarray, spacing: float) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Create empty grid based on point cloud extent and spacing.
    
    Args:
        points: Array of xyz coordinates
        spacing: Desired grid cell size
        
    Returns:
        Tuple containing:
        - 2D numpy array initialized with NaN
        - Tuple of grid extents (xmin, xmax, ymin, ymax)
    """
    # Get point cloud extent
    xmin, ymin = np.min(points[:, :2], axis=0)
    xmax, ymax = np.max(points[:, :2], axis=0)
    
    # Add small buffer
    buffer = spacing
    xmin -= buffer
    xmax += buffer
    ymin -= buffer
    ymax += buffer
    
    # Calculate grid dimensions
    nx = int(np.ceil((xmax - xmin) / spacing))
    ny = int(np.ceil((ymax - ymin) / spacing))
    
    # Create empty grid
    grid = np.full((ny, nx), np.nan)
    
    logging.info(f"Created grid with dimensions {ny} x {nx}")
    return grid, (xmin, xmax, ymin, ymax)

def check_triangle_edges(vertices: np.ndarray, max_length: float) -> bool:
    """
    Check if any edge of a triangle exceeds maximum allowed length.
    
    Args:
        vertices: Array of shape (3,2) containing triangle vertex coordinates
        max_length: Maximum allowed edge length
        
    Returns:
        bool: True if any edge exceeds max_length
    """
    edges = [
        np.linalg.norm(vertices[1] - vertices[0]),
        np.linalg.norm(vertices[2] - vertices[1]),
        np.linalg.norm(vertices[0] - vertices[2])
    ]
    return any(edge > max_length for edge in edges)

def visualize_delaunay(points, tri):
    plt.triplot(points[:, 0], points[:, 1], tri.simplices, color='gray', alpha=0.5)
    plt.scatter(points[:, 0], points[:, 1], color='red', s=10, label='Input Points')
    plt.title("Delaunay Triangulation")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()


def calculate_barycentric_coordinates(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    """
    Calculate barycentric coordinates of a point within a triangle.
    
    Args:
        point: Array of shape (2,) containing xy coordinates
        triangle: Array of shape (3,2) containing triangle vertex coordinates
        
    Returns:
        numpy.ndarray: Array of shape (3,) containing barycentric coordinates
    """
    v0 = triangle[2] - triangle[0]
    v1 = triangle[1] - triangle[0]
    v2 = point - triangle[0]
    
    # Calculate dot products
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    
    # Calculate barycentric coordinates
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    
    return np.array([u, v, w])

def interpolate_grid(points: np.ndarray, grid: np.ndarray, 
                    extent: Tuple[float, float, float, float],
                    spacing: float, max_edge_length: float) -> np.ndarray:
    """
    Interpolate grid values using Delaunay triangulation.
    
    Args:
        points: Array of xyz coordinates
        grid: Empty grid initialized with NaN
        extent: Tuple of grid extents (xmin, xmax, ymin, ymax)
        spacing: Grid cell size
        max_edge_length: Maximum allowed triangle edge length
        
    Returns:
        numpy.ndarray: Grid with interpolated values and NaN for invalid areas
    """
    # Create Delaunay triangulation
    tri = Delaunay(points[:, :2])
    logging.info(f"Created triangulation with {len(tri.simplices)} triangles")
    visualize_delaunay(points, tri)
    
    # Get grid coordinates
    xmin, xmax, ymin, ymax = extent
    ny, nx = grid.shape
    x = np.linspace(xmin, xmin + spacing * (nx-1), nx)
    y = np.linspace(ymin, ymin + spacing * (ny-1), ny)
    xx, yy = np.meshgrid(x, y)
    
    # For each grid point
    for i in range(ny):
        for j in range(nx):
            point = np.array([xx[i,j], yy[i,j]])
            
            # Find containing triangle
            simplex = tri.find_simplex(point)
            
            if simplex >= 0:  # Point is inside triangulation
                vertices = tri.simplices[simplex]
                triangle_coords = points[vertices, :2]
                
                # Check triangle edge lengths
                if check_triangle_edges(triangle_coords, max_edge_length):
                    continue  # Leave as NaN (NoData cell)
                    
                # Calculate barycentric coordinates
                barycentric = calculate_barycentric_coordinates(point, triangle_coords)
                
                # Interpolate z value
                grid[i,j] = np.sum(barycentric * points[vertices, 2])
                
        if i % 20 == 0:  # Progress update
            logging.info(f"Processed row {i}/{ny}")
            
    return grid

def visualize_grid(grid: np.ndarray, points: np.ndarray = None, title: str = "Interpolated Grid") -> None:
    """
    Visualize the interpolated grid.
    
    Args:
        grid: 2D array containing interpolated values and NaN
        points: Optional original xyz points to overlay
        title: Plot title
    """
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot interpolated grid
    im = plt.imshow(grid, origin='lower', cmap='viridis')
    plt.colorbar(im, label='Z Value')
    
    # Add original points if provided
    if points is not None:
        plt.scatter(points[:,0], points[:,1], c='black', 
                   s=20, alpha=0.5, label='Original Points')
    
    # Add grid lines
    plt.grid(True, color='white', linestyle='-', alpha=0.2)
    
    # Add statistics
    valid_cells = np.count_nonzero(~np.isnan(grid))
    total_cells = grid.size
    coverage = (valid_cells / total_cells) * 100
    
    stats_text = f'Grid Coverage: {coverage:.1f}%\n'
    stats_text += f'Valid Cells: {valid_cells}/{total_cells}\n'
    if np.any(~np.isnan(grid)):
        stats_text += f'Z Range: [{np.nanmin(grid):.3f}, {np.nanmax(grid):.3f}]'
    
    plt.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top', fontsize=9)
    
    # Add labels and title
    plt.title(title)
    plt.xlabel('X Grid Cell')
    plt.ylabel('Y Grid Cell')
    
    plt.tight_layout()
    plt.show()

def main(input_file: str, grid_spacing: float, max_edge_length: float) -> None:
    """
    Main execution function.
    
    Args:
        input_file: Path to input xyz data file
        grid_spacing: Desired grid cell size
        max_edge_length: Maximum allowed triangle edge length
    """
    # Verify input file exists
    if not os.path.exists(input_file):
        logging.error(f"Input file not found: {input_file}")
        return
    
    # Read and print first few lines of input file
    with open(input_file, 'r') as f:
        print("\nFirst few lines of input file:")
        for i, line in enumerate(f):
            if i < 5:  # Print first 5 lines
                print(f"Line {i}: {line.strip()}")
            else:
                break
    
    # Read input data
    logging.info(f"Attempting to read file: {input_file}")
    points = read_xyz_data(input_file)
    
    if points is None:
        logging.error("Failed to read input data")
        return
        
    # Setup grid
    logging.info(f"Setting up grid with spacing: {grid_spacing}")
    grid, extent = setup_grid(points, grid_spacing)
    
    # Perform interpolation
    logging.info(f"Starting interpolation with max edge length: {max_edge_length}")
    grid = interpolate_grid(points, grid, extent, grid_spacing, max_edge_length)
    
    # Calculate coverage statistics
    valid_cells = np.count_nonzero(~np.isnan(grid))
    total_cells = grid.size
    coverage = (valid_cells / total_cells) * 100
    
    logging.info(f"Grid coverage: {coverage:.1f}% ({valid_cells}/{total_cells} cells)")
    
    # Visualize results
    visualize_grid(grid, points)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    INPUT_FILE = "test_data/case2_scattered_points.txt"
    GRID_SPACING = 1.0  # Cell size for output grid
    MAX_EDGE_LENGTH = 20.0  # Maximum allowed triangle edge length
    
    logging.info("Starting program with:")
    logging.info(f"Input file: {INPUT_FILE}")
    logging.info(f"Grid spacing: {GRID_SPACING}")
    logging.info(f"Max edge length: {MAX_EDGE_LENGTH}")
    
    try:
        main(INPUT_FILE, GRID_SPACING, MAX_EDGE_LENGTH)
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
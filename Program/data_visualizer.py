# Test Data Visualizer
import os
from test_data_generator import create_all_test_cases
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_test_case_name(filename):
    """Extract a clean test case name from filename"""
    # Remove extension and path
    base_name = os.path.splitext(os.path.basename(filename))[0]
    # Convert to title case and replace underscores
    return base_name.replace('_', ' ').title()

def visualize_xyz_data(points, filename, show_raw_filename=False):
    """
    Create three visualizations of the input points:
    1. 2D scatter plot with z-values as colors
    2. 3D surface plot
    3. Point density plot
    
    Args:
        points: numpy array of shape (n,3) containing xyz coordinates
        filename: Name of the test file
        show_raw_filename: If True, shows raw filename, otherwise formats it
    """
    # Create figure with three subplots
    fig = plt.figure(figsize=(15, 5))
    
    # Format title based on filename
    if show_raw_filename:
        title = f"Test Data: {filename}"
    else:
        title = f"Test Case: {get_test_case_name(filename)}"
    
    # 1. 2D scatter plot with z-values as colors
    ax1 = fig.add_subplot(131)
    scatter = ax1.scatter(points[:,0], points[:,1], c=points[:,2], 
                         cmap='viridis', s=30)
    plt.colorbar(scatter, ax=ax1, label='Z Value')
    ax1.set_title('2D View (Color = Z)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True)
    
    # 2. 3D surface plot
    ax2 = fig.add_subplot(132, projection='3d')
    scatter3d = ax2.scatter(points[:,0], points[:,1], points[:,2],
                           c=points[:,2], cmap='viridis')
    plt.colorbar(scatter3d, ax=ax2, label='Z Value')
    ax2.set_title('3D View')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # 3. Point density plot
    ax3 = fig.add_subplot(133)
    density = ax3.hist2d(points[:,0], points[:,1], bins=20, cmap='YlOrRd')
    plt.colorbar(density[3], ax=ax3, label='Point Count')
    ax3.set_title('Point Density')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    
    # Add main title and file info
    plt.suptitle(title, fontsize=16, y=1.05)
    
    # Add file info in small text
    file_info = f"File: {filename}\nPoints: {len(points)}"
    fig.text(0.99, 0.02, file_info, fontsize=8, 
             horizontalalignment='right', verticalalignment='bottom',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.tight_layout()
    return fig

def analyze_point_distribution(points, filename):
    """
    Print statistical analysis of point distribution
    
    Args:
        points: numpy array of shape (n,3) containing xyz coordinates
        filename: Name of the test file
    """
    print(f"\nAnalysis for: {filename}")
    print("=" * 50)
    
    # Basic statistics
    print(f"Total number of points: {len(points)}")
    
    # Coordinate ranges
    x_range = np.ptp(points[:,0])
    y_range = np.ptp(points[:,1])
    z_range = np.ptp(points[:,2])
    print(f"\nCoordinate Ranges:")
    print(f"X range: {x_range:.2f}")
    print(f"Y range: {y_range:.2f}")
    print(f"Z range: {z_range:.2f}")
    
    # Point density
    area = x_range * y_range
    density = len(points) / area
    print(f"\nAverage point density: {density:.2f} points per square unit")
    
    # Z-value distribution
    print(f"\nZ-value Statistics:")
    print(f"Mean Z: {np.mean(points[:,2]):.2f}")
    print(f"Std Z: {np.std(points[:,2]):.2f}")
    print(f"Min Z: {np.min(points[:,2]):.2f}")
    print(f"Max Z: {np.max(points[:,2]):.2f}")
    print("-" * 50)

def main():
    # Create test data
    print("Generating test data...")
    create_all_test_cases()
    
    # Get list of test files
    test_dir = "test_data"
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.txt')])
    
    print(f"\nFound {len(test_files)} test files:")
    for f in test_files:
        print(f"- {f}")
    
    # Visualize each test file
    for filename in test_files:
        filepath = os.path.join(test_dir, filename)
        print(f"\nProcessing: {filename}")
        print("=" * 50)
        
        try:
            # Load data, skipping header row
            points = np.loadtxt(filepath, skiprows=1)
            
            # Create visualization
            fig = visualize_xyz_data(points, filename)
            
            # Print analysis
            analyze_point_distribution(points, filename)
            
            # Show plot (will pause here until window is closed)
            plt.show()
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
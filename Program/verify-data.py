# verify_data.py
import numpy as np

def verify_test_file(filename):
    """Verify the content of a test data file"""
    print(f"\nVerifying file: {filename}")
    print("-" * 50)
    
    # Read raw file content
    print("Raw file content (first 5 lines):")
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if i < 5:
                print(f"Line {i}: {line.strip()}")
            else:
                break
    
    # Try reading with numpy
    print("\nReading with numpy.loadtxt:")
    try:
        # Try without header
        data = np.loadtxt(filename)
        print(f"Shape without skipping header: {data.shape}")
    except:
        print("Failed to read without skipping header")
        
        try:
            # Try with header
            data = np.loadtxt(filename, skiprows=1)
            print(f"Shape with skipping header: {data.shape}")
            
            # Check for non-finite values
            non_finite = ~np.isfinite(data)
            if np.any(non_finite):
                print("\nFound non-finite values:")
                rows, cols = np.where(non_finite)
                for r, c in zip(rows, cols):
                    print(f"Row {r}, Column {c}: {data[r, c]}")
            else:
                print("\nAll values are finite")
            
            # Show data statistics
            print("\nData statistics:")
            print(f"Min values: {np.min(data, axis=0)}")
            print(f"Max values: {np.max(data, axis=0)}")
            print(f"Mean values: {np.mean(data, axis=0)}")
            
            # Show first few rows
            print("\nFirst few rows:")
            print(data[:5])
            
        except Exception as e:
            print(f"Failed to read with header: {str(e)}")

if __name__ == "__main__":
    filename = "test_data/case1_regular_grid.txt"
    verify_test_file(filename)
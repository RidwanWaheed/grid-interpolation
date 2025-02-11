# Grid Interpolation using Delaunay Triangulation

This project implements a solution for converting irregular point measurements into a regular grid while handling areas with insufficient data coverage. The implementation uses Delaunay triangulation for interpolation and employs careful handling of areas with insufficient data coverage.

## Project Structure

- `xyz_grid_interpolation.py`: Main implementation of the grid interpolation
- `verify_data.py`: Data verification and validation script
- `test_data_visualizer.py`: Tools for data visualization
- `test_data/`: Directory containing test datasets

## Requirements

- Python 3.x
- NumPy
- SciPy
- Matplotlib

## Installation

```bash
pip install numpy scipy matplotlib
```

## Usage

```bash
python xyz_grid_interpolation.py
```

Key parameters:
- GRID_SPACING: Controls output grid resolution (default: 1.0)
- MAX_EDGE_LENGTH: Maximum allowed triangle edge length (default: 20.0)

## Features

- Data verification and validation
- Initial data visualization
- Delaunay triangulation-based interpolation
- NoData cell handling
- Result visualization

## Documentation

See the technical report for detailed information about:
- Methodology
- Implementation details
- Results analysis
- Parameter selection guidelines

## Author

Ridwan Waheed  
TU Dresden

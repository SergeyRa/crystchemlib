# *crystchemlib* v.0.9.3
Python library for crystallographic and crystal chemical analysis with *streamlit* GUI

## Maintainer
Sergey V. Rashchenko (rashchenkos@gmail.com)

## v.0.9.3 features
* Installation scripts
* Handling of incommensurate structures
* Birch-Murnaghan equations of state fitting
* GUI for merging CIFs with insertion of custom keys (e.g. for pressure)
* Visualization of nets in 'Dense atomic nets' module
* Module for visualization of Voronoi-Dirichlet polyhedra
* Parametric analysis of Voronoi-Dirichlet volumes in the main GUI

## Installation

### Windows 10
1. Download and install Python for Windows from https://apps.microsoft.com/detail/9PNRBTZXMB4Z
2. Download *crystchemlib* files from https://github.com/SergeyRa/crystchemlib (Code - Download ZIP) and extract crystchemlib-main folder to a directory of your choice (should be accessible  for non-administrator users)
3. Launch Win_install.bat for installation (this will install necessary Python libraries)
4. Launch Win_run.bat to open GUI (for the first time you may be asked to provide optional e-mail - just leave it blank and press Enter)

### Linux
1. Download *crystchemlib* files from https://github.com/SergeyRa/crystchemlib (Code - Download ZIP) and extract crystchemlib-main folder to a directory of your choice
2. Launch Linux_install.sh for installation (this will install necessary Python libraries)
3. Launch Linux_run.sh to open GUI (for the first time you may be asked to provide optional e-mail - just leave it blank and press Enter)

## Manual
A brief manual is available in *J. Appl. Cryst.* paper (see author manuscript in JAC2025.pdf). The algorithm of dense plain nets searching is described in *J. Struct. Chem.* paper (see author manuscript in JSC2025.pdf).

## Citing
If you use *crystchemlib* in any part of your project, please cite it in your publications: S.V. Rashchenko (2025). *J. Appl. Cryst.* **58**, https://doi.org/10.1107/S1600576724011956.

## Known limitations and bugs
* Estimated standard deviations (esd) of bond lengths and angles are calculated without symmetry constraints
* The use of GUI in a browser instance with a large number of open tabs may result in restricted access to memory and slow work

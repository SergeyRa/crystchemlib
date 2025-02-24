# *crystchemlib*
Python library for crystallographic and crystal chemical analysis with *streamlit* GUI

## Maintainer
Sergey V. Rashchenko (rashchenkos@gmail.com)

## Installation

### Windows
1. Download and install Anaconda Python environment from https://www.anaconda.com/download/success
2. Launch Anaconda Prompt shell from Windows Start menu ("Run as Administrator" may be required)
3. Install necessary packages by executing "conda install matplotlib numpy pandas plotly scipy streamlit sympy" command in Anaconda Prompt
4. Download *crystchemlib* files from https://github.com/SergeyRa/crystchemlib (Code - Download ZIP) and extract crystchemlib-main folder into your "C:\\\Users\YOUR_USERNAME" folder
5. Launch graphical user interface (GUI) by executing "streamlit run crystchemlib-main/crystchemlibGUI.py" command in Anaconda Prompt

## Manual
A brief manual is available in *J. Appl. Cryst.* paper (see author manuscript in JAC2025.pdf)

## Citing
If you use *crystchemlib* in any part of your project, please cite it in your publications: S.V. Rashchenko (2025). *J. Appl. Cryst.* **58**, https://doi.org/10.1107/S1600576724011956.

## Known limitations and bugs
* Estimated standard deviations (esd) of bond lengths and angles are calculated without symmetry constraints
* The use of GUI in a browser instance with a large number of open tabs may result in restricted access to memory and slow work

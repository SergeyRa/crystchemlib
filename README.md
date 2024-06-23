# crystchemlib
Python library for crystallographic and crystal chemical analysis with streamlit GUI. Contact author at rashchenkos@gmail.com for bug reports, questions and suggestions.

## Installation

### Windows

1. Download and install Anaconda Python environment from https://www.anaconda.com/download/success
2. Launch Anaconda Prompt shell from Windows Start menu ("Run as Administrator" may be required)
3. Install streamlit and plotly packages by executing "conda install streamlit plotly" command in Anaconda Prompt
4. Download crystchemlib files from https://github.com/SergeyRa/crystchemlib (Code - Download ZIP) and extract crystchemlib-main folder into your "C:\Users\YOUR_USERNAME" folder
5. Launch graphical user interface (GUI) by executing "streamlit run crystchemlib-main/crystchemlibGUI.py" command in Anaconda Prompt

## Known limitations and bugs
* Estimated standard deviations (esd) of bond lengths and angles are calculated without symmetry constraints
* The use of GUI in a browser instance with a large number of open tabs may result in restricted access to memory and slow work

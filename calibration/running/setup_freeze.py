from cx_Freeze import setup, Executable

#import sys

VERSION = '1.0.1 19/12/2018'

long_description = """\
Model design and processes included
===================================
The Community Water Model (CWATM) will be designed for the purpose to assess water
availability, water demand and environmental needs. It includes an accounting of how
future water demands will evolve in response to socioeconomic change and how water 
availability will change in response to climate.

Next-generation global hydro-economic modeling framework
========================================================
The Community Water Model will help to develop a next-generation hydro-economic 
modeling tool that represents the economic trade-offs among water supply 
technologies and demands. The tool will track water use from all sectors and will
identify the least-cost solutions for meeting future water demands under policy 
constraints.  

Contact CWAT
============
http://www.iiasa.ac.at/cwatm 
"""

# Dependencies are automatically detected, but it might need
# fine tuning. "netcdftime"
buildOptions = dict(packages = ["numpy","netCDF4","cftime","osgeo","scipy","flopy"], excludes = ["scipy.spatial.cKDTree"],
     include_files = ["README.md","cwatm/metaNetcdf.xml","LICENSE.txt","settings1.ini"])
	 
#"hydrological_modules/routing_reservoirs/t5.dll","hydrological_modules/routing_reservoirs/t5_linux.so"

base = 'Console'

# GUI applications require a different base on Windows (the default is for a
# console application).
#base = None
#if sys.platform == "win32":
#   base = "Win32GUI"
#print (sys.platform)


executables = [
    Executable('run_cwatm.py', base=base, targetName = 'cwatm.exe')
]

setup(	
    name = "Community Water Model - CWATM",
    version = VERSION,
    description = "Hydrological model",
    author = "IIASA Water Program - Peter Burek et al.",
    author_email = "burek@iiasa.ac.at",
    url = "http://www.iiasa.ac.at/cwatm",
    download_url = "https://github.com/CWatM/",
    keywords = ["hydrology", "global", "model"],
    long_description=long_description,

    options = dict(build_exe = buildOptions),
    executables = executables)

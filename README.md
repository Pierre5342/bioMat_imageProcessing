# bioMat_imageProcessing

This script has been developed by Pierre MAINDRON as a Master Project realised in the BioMat Department od the "Centre Ingénierie Santé" of Ecole des Mines de St-Etienne (Mines Saint-Etienne, Université Jean Monnet, INSERM, U1059 SAINBIOSE, 42023 Saint-Etienne France), last update: 09/27/2024. Its role is to perform automated nuclei segmentation and analysis on 3D biological images.
The images used to develop and test this algorithm come from a set of 3D images (stacks of 2D images) acquired using a two-photon confocal microscope, 4 h after cell seeding in a bioceramic scaffold. The nuclei and the cells’ membrane are observed using respectively DAPI and Dil to dye them and fluorescence microscopy to observe them.
This script takes the .nd2 files produced by the microscope, performs the segmentation and analysis of the nuclei and returns the segmented image, an excel file containing various data, some graphs.


Environment

The image processing script was developed with Python 3.11.7, and requires several python modules: 
  •	sys, getopt et subprocess: reading of the command line and execution of bash commands
  •	nd2 : nd2 files and metadata reading
  •	numpy: matrix (thus images) manipulation
  •	pandas: data manipulation, excel file generation and writing
  •	cv2, scikit-image, scipy: image processing operations
  •	scikit-learn : mathematical operations
  •	matplotlib: graph generation
  •	time, math
This version of the script can be run only on Windows operating systems and does not require a GPU.


Global Process

The script includes two Python files: one executable file and another file containing all the auxiliary functions called in the executable.
The executable file is divided into 3 major parts: reading of the command line and initialisation of the variables, the print_help function, the main function.
The first part of the code is used to read the command line, identify the chosen options and input parameters, and initialise the variables consequently. If mandatory parameters are missing or the files format are not the one expected, error messages are displayed and the script execution is stopped. 
If the help option is used, the print_help function is called and the rest of the script is not executed.
Otherwise, the main function, itself subdivided in two major parts, is called. The first part is dedicated to preliminary work: retrieving the name of the images to analyse (if a directory name is input), reconstituting the entire path of the image(s) to analyse and store them in a list, defining the name of and creating the output directory, initialising the excel file. The second part is dedicated to the image segmentation and analysis itself. 


Tutorial

The script is executed via the following command line:
	python   nuclei_analysis_J0.py   [-f fileName | -d directoryName]   [-o outputName]   [-h]
The parameters that can be used for the script execution are the following ones.
Mandatory parameters:
        -f | --file <fileName>: name of the image to analyse (.nd2 image format is the only one accepted by the script).
                 OR
        -d | --directory <directoryName>: if you want to analyze several images at once, name of the folder in which the images to be analyzed are stored (only .nd2 files present in the folder will be considered).
Optional parameters:
        -o | --output-directory <outputDirName>: name of the folder in which the results produced by the script will be stored (segmented image(s), excel file containing analysis data, graphs). If the folder doesn't exist, it will be created; if it does, a warning is issued to warn of the potential overwriting of existing files with identical names.
        -h | --help: displays this documentation.

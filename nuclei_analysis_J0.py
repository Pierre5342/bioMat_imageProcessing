#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 12:07:48 2024

@author: pierre.maindron
"""

import sys, getopt, subprocess
import nd2, math, time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import pi
from skimage import measure, io
from scipy import ndimage

from auxiliary_functions import rescale, crop, find_groove_angle_using_fileName, correct_contrasts
from auxiliary_functions import otsu_binaryWatershed
from auxiliary_functions import membranes_binarisation, find_intersect_points, find_rotation_angle, find_axis_origins
from auxiliary_functions import distance, count_contacts, detect_groups
    





def main(fileName, dirName, output_dirName) :
    
    # Add all images' filepath to a list
    filepath_list = []
    
    if fileName != "NA" :
        path_to_directory = subprocess.run(["cd"], capture_output=True, shell=True, text=True).stdout
        directory_path = path_to_directory[:-1]
        
        filepath = '/'.join([directory_path, fileName])
        filepath_list.append(filepath)
        
        excelName = "data_" + fileName.split("/")[-1][:-4] + ".xlsx"
        
        if output_dirName == "NA":
            actual_fileName = fileName.split("/")
            output_dirName = "analysisResults_" + actual_fileName[-1][:-4]
    
    if dirName != "NA" :
        try :
            files = subprocess.run(["dir", "/b", dirName+"\*.nd2"], capture_output=True, shell=True, text=True)
        except FileNotFoundError :
            print("This directory (" + dirName +") does not exist")
            sys.exit(1)
        
        path_to_directory = subprocess.run(["cd"], capture_output=True, shell=True, text=True).stdout
        directory_path = path_to_directory[:-1]
        
        files_names = files.stdout.split("\n")
        
        for name in files_names[:-1] :
            
            index = name.rfind('œ')
            nliste = list(name)
            nliste[index] = '£'
            name = ''.join(nliste)
            
            filepath = '/'.join([directory_path, dirName, name])
            filepath_list.append(filepath)
        
        excelName = "data_" + dirName + ".xlsx"
        
        if output_dirName == "NA":
            output_dirName = "analysisResults_" + dirName
    
    
    # Analyse all images
    # Store segmented images in outputDirName and the data in an excel file
    
    directories = subprocess.check_output(["dir", "/ad", "/b"], shell=True, text=True)
    existing_dir = directories.split("\n")
    
    for i, name in enumerate(existing_dir[:-1]) :
        index = name.rfind('œ')
        if index != -1 :
            nliste = list(name)
            nliste[index] = '£'
            name = ''.join(nliste)
            existing_dir[i] = name
            
    if output_dirName in existing_dir :
        val = input("A directory named "+output_dirName+" already exists, its content may be overwritten. Do you want to continue ? (y/n)")
        
        if val != "y" and val != "Y" :
            sys.exit(1)
    else :
        subprocess.run(["mkdir", output_dirName], shell=True)

    
    with pd.ExcelWriter(output_dirName+"/"+excelName) as writer:
        
        for k, filepath in enumerate(filepath_list) :
            t1 = time.time()
            ######################### Image pre-treatment #########################
            
            ## Retrieving the voxels' sizes in the image metadata
            with nd2.ND2File(filepath) as ndfile:
                voxels_size = ndfile.voxel_size()
                ndfile.close()
                voxel_size_xy = voxels_size[0]
                voxel_size_z = voxels_size[2]
            
            ## Reading the image
            data_img = nd2.imread(filepath)
            
            ## Rescaling and cropping of the image
            rescaled_nuclei, rescaled_membranes = crop(rescale(data_img, voxel_size_xy, voxel_size_z))
            
            ## Retrieving the groove's angle and correcting contrasts if necessary
            ## (in the case of 180° grooves)
            angle = find_groove_angle_using_fileName(filepath)
            
            if angle == pi :
                rescaled_nuclei = correct_contrasts(rescaled_nuclei)
            
            ######################### Axis definition #########################
            
            ## Membranes channel image binarization
            
            binary_membranes = membranes_binarisation(rescaled_membranes)
            
            ## Find the position of the groove in the images set of coordinates
            ## (rotation and translation) to define a new set of axis
            
            # Find Phi
            intersection_points = find_intersect_points(binary_membranes, angle)
            
            phi = find_rotation_angle(intersection_points[:,1], intersection_points[:,0])
            
            # Apply a rotation of angle phi on the og membranes image + binarization
            rotated_membranes = ndimage.rotate(rescaled_membranes, math.degrees(phi), axes=(0,1))
    
            binary_rotated_membranes = membranes_binarisation(rotated_membranes)
            
            # Find Theta
            intersection_points = find_intersect_points(binary_rotated_membranes, angle)
            
            theta = find_rotation_angle(intersection_points[:,1], intersection_points[:,2])
            
            # Apply a rotation of angle theta on the og image + binarization
            double_rotated_membranes = ndimage.rotate(rotated_membranes, -math.degrees(theta), axes=(1,2))
            
            binary_drm = membranes_binarisation(double_rotated_membranes)
            
            # Getting x0 and z0
            intersection_points = find_intersect_points(binary_drm, angle)
            
            z0, x0 = find_axis_origins(intersection_points)
            
            
            ######################## Nuclei segmentation ######################
            rotated_nuclei = ndimage.rotate(rescaled_nuclei, math.degrees(phi), axes=(0,1))
            
            double_rotated_nuclei = ndimage.rotate(rotated_nuclei, -math.degrees(theta), axes=(1,2))
            
            segmented_nuclei = otsu_binaryWatershed(double_rotated_nuclei)
            
            path_list = filepath.split("/")
            
            img_name = path_list[-1].split(".")
            
            io.imsave(output_dirName + "/segmented_" + img_name[0] + ".tif", segmented_nuclei, check_contrast=False)
            
            
            ######################### Nuclei analysis #########################
            volumes = np.unique(segmented_nuclei, return_counts=True)[1]
            
            nb_nuclei = int(len(volumes)-1)
            
            regions = measure.regionprops(segmented_nuclei, double_rotated_nuclei)
    
            centroids = np.empty((nb_nuclei,5))
            
            nuclei_distances = np.zeros((nb_nuclei+1,nb_nuclei+1))
            
            mean_eq_diameter = 0
    
            for i, vol in enumerate(volumes[1:]):
                ## Individual nuclei caracteristics
                
                z, y, x = regions[i].centroid
                
                (z_min, y_min, x_min, z_max, y_max, x_max) = regions[i].bbox
                
                nuclei = segmented_nuclei == i+1
                
                nucleus = nuclei[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
                
                mesh_nuclei = measure.marching_cubes(nucleus)
                
                nuclei_surface = measure.mesh_surface_area(mesh_nuclei[0], mesh_nuclei[1])
                
                equivalent_diameter = regions[i].equivalent_diameter_area
                
                phi_v = (pi * (equivalent_diameter**2)) / nuclei_surface
                
                centroids[i, :] = (x-x0, y, z-z0, vol, phi_v)
                
                ## Distances between close nuclei
                for j in range(1,i+1):
                    nuclei_distances[i+1,j] = distance(centroids[i,:3], centroids[j-1,:3])
                
                mean_eq_diameter += equivalent_diameter
                    
            
            tuples = [(img_name[0], "", "Coordinates", "X"), 
                      (img_name[0], "", "Coordinates", "Y"), 
                      (img_name[0], "", "Coordinates", "Z"), 
                      (img_name[0], "", "Volume", "µm³"),
                      (img_name[0], "", "Sphericity", "factor")]
            
            ind_col = pd.MultiIndex.from_tuples(tuples)
    
            ind_row = pd.RangeIndex(1, nb_nuclei+1)
            
            nuclei_data = pd.DataFrame(data=centroids, columns=ind_col, index=ind_row)
            
            if k < 9 :
                sheet_name = "img_000"+str(k+1)
            else :
                sheet_name = "img_00"+str(k+1)
                
            nuclei_data.to_excel(writer, sheet_name=sheet_name, index_label= "Label")
            
            
            ## Distances between close nuclei
            mean_eq_diameter = mean_eq_diameter/nb_nuclei
            
            values, indexes = [], []
            for i in range(1, nb_nuclei+1):
                for j in range(1, i) :
                    dist_ij = nuclei_distances[i,j]
                    if dist_ij != 0 and dist_ij <= 2*mean_eq_diameter :
                        values.append(dist_ij)
                        indexes.append((i,j))
            
            ind_row = pd.MultiIndex.from_tuples(indexes)
            
            distances_data = pd.DataFrame(data=values, columns=["Distance between the nuclei (µm)"], index=ind_row)
            
            distances_data.to_excel(writer, sheet_name=sheet_name, index_label=["Ref. nucleus", "Close nuclei"], startcol=8, startrow=4)
            
            
            ## Nuclei density
            props = measure.regionprops(binary_drm)
            
            b_box = props[0].bbox
            
            sorted_centroids = np.sort(centroids[:,2])
            density, interval = [], 50
            max_hight = int(interval* (np.max(sorted_centroids)//interval + 1))
            p = 0
            
            if angle == pi :
                area = interval * (b_box[3] - b_box[0]) * 2
            else :
                area = (interval / math.cos(angle/2)) * (b_box[3] - b_box[0]) * 2
                
            for i in range(0,max_hight,interval) :
                j = 0
                while sorted_centroids[p] < i :
                    p += 1
                    j += 1
                
                if angle == pi and i == 0 :
                    density.append(j / (area + 250*(b_box[3] - b_box[0])))
                else :
                    density.append(j/(area*10**(-8)))
            
            ind_row = [str(i)+" - "+str(i+interval) for i in range(0, max_hight, interval)] 
            
            density_data = pd.DataFrame(data=np.array(density), columns=["Nuclei density (Number of nuclei/cm^2)"], index=ind_row)
            
            density_data.to_excel(writer, sheet_name=sheet_name, index_label=["Intervals along Z axis"], startcol=13, startrow=4)
            
            
            # Graphs creation
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(10, 5))
            
            x = [i for i in range(0, max_hight, interval)]
            
            ax2.barh(x, density, height=interval)
            ax1.scatter(centroids[:,0], centroids[:,2], s=volumes[1:]/100, alpha=0.4)
            
            ax1.axis('equal')
            ax1.grid(visible=True, linestyle='--')
            ax2.grid(visible=True, axis='x', linestyle='--')
            ax1.set_ylim([0, 10*math.ceil(np.max(centroids[:,2])/10)+10])
            
            ax1.set_title("Nuclei centroids")
            ax2.set_title("Nuclei density along the vertical axis")
            ax1.set_ylabel('Z')
            ax1.set_xlabel('X')
            ax2.set_xlabel('Nuclei density (Number of nuclei/cm²)')
            fig.savefig(output_dirName + "/density_" + img_name[0] + ".tif")
            fig.clear()
            
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(projection='3d')
            ax.scatter(centroids[:,0], centroids[:,1], centroids[:,2], s=volumes[1:]/20)
            ax.set_xlabel("X (Goove's width)")
            ax.set_ylabel("Y (Groove's length)")
            ax.set_zlabel("Z (Groove's height)")
            fig.savefig(output_dirName + "/centroids_" + img_name[0] + ".tif")
            fig.clear()
            
            
            
            
            ## Contacts and general information (number of nuclei, mean volume)
            
            mean_vol = np.mean(np.array(volumes[1:]))
            
            nb_contacts = count_contacts(segmented_nuclei)
            
            general_data = pd.DataFrame(data=np.array([nb_nuclei, mean_vol, nb_contacts]), index=["Number of nuclei", "Mean volume", "Number of contacts"]).T
            
            general_data.to_excel(writer, sheet_name=sheet_name, startcol=8)
            
            groups = detect_groups(segmented_nuclei)
            
            ind_row = pd.RangeIndex(1, len(groups)+1)
            contacts_data = pd.DataFrame(data=np.array(groups), columns=["Numbers of groups"], index=ind_row)
            
            contacts_data.to_excel(writer, sheet_name=sheet_name, index_label=["Group size (nb of connected nuclei)"], startcol=17, startrow=4)
            
            
            t2 = time.time()
            t_min = round((t2-t1)//60)
            t_s = round((t2-t1)%60)
            print("Image n°"+str(k+1)+" analysed in "+str(t_min)+"min "+str(t_s)+"s")
            
        
        


def print_help():
    print("\n")
    print("This script is designed to use the raw image coming directly from the confocal microscope, and perform all the necessary operation to segment and analyse the nuclei, and save the segmented images, an excel file containing the data from the image analysis and several graphs.")
    print("The script is executed via the following command line:")
    print("\t python nuclei_analysis_J0.py [-f fileName | -d directoryName] [-o outputName] [-h]")
    print("\n")
    print("The parameters that can be used for the script execution are the following ones.")
    print("Mandatory parameters:")
    print("\t-f | --file <fileName> : name of the image to analyse (.nd2 image format is the only one accepted by the script).")
    print("\t\t OR")
    print("\t-d | --directory <directoryName> : if you want to analyze several images at once, name of the folder in which the images to be analyzed are stored (only .nd2 files present in the folder will be considered).")
    print("\n")
    print("Optionnal parameters:")
    print("\t-o | --output-directory <outputDirName> : name of the folder in which the results produced by the script will be stored (segmented image(s), excel file containing analysis data, graphs). If the folder doesn't exist, it will be created; if it does, a warning is issued to warn of the potential overwriting of existing files with identical names.")
    print("\t-h | --help : displays this documentation.")
    print("\n")







if __name__ == "__main__":

    fileName = "NA"
    dirName = "NA"
    output_dirName = "NA"
    
    # Get parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hf:d:o:", ["help", "file", "directory", "output-directory"])
    except getopt.GetoptError:
        print("Unable to parse the arguments\n", file=sys.stderr)
        sys.exit(1)

    for opt, arg in opts:
        if opt == "-h" or opt == "--help":
            print_help()
            sys.exit(0)
        
        if opt == "-f" or opt == "file":
            fileName = arg
            if fileName[-4:] != ".nd2" :
                print("Wrong image file format, the input image must be an nd2 file (.nd2)")
                print(fileName)
                sys.exit(1)

        if opt == "-d" or opt == "directory":
            dirName = arg
            
        if opt == "-o" or opt == "output-directory":
            output_dirName = arg

    if fileName == "NA" and dirName == "NA":
            print("ERROR: the -f or -d options are mandatory")
            sys.exit(1)
            
    if output_dirName == "NA":
        print("WARNING: the use of the -o option is recommended")

    
    main(fileName, dirName, output_dirName)
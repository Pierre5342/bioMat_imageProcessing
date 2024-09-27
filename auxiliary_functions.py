# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 14:45:30 2024

@author: pierre.maindron
"""

import cv2, math

import numpy as np

from math import pi
from scipy import ndimage
from sklearn.linear_model import LinearRegression
from skimage import transform, morphology, filters, segmentation, feature, measure


############################# Image pre-treatment #############################

## Images rescaling
def rescale(data, voxel_size_xy, voxel_size_z):
    
    """
    Rescale the given image to return an isotropic one.
    
    Parameters
    ----------
    data : numpy.array
        Non isotropic 3D image (containing at least 2 channels)
    voxel_size_xy : Float
        Pixel width.
    voxel_size_z : Float
        Voxel depth.

    Returns
    -------
    rescaled : numpy.array
        Rescaled, isotropic image.

    """    
    (z_size, channels, y_size, x_size) = data.shape
        
    shape = (int(round(z_size * voxel_size_z)), channels, int(round(y_size * voxel_size_xy)), int(round(x_size * voxel_size_xy)))
    
    rescaled = transform.resize(data,shape)
    
    return rescaled


## Image cropping
def crop(data):
    
    """
    Crop the image along the X axis to reduce its size and save computation time

    Parameters
    ----------
    data : numpy.array
        3D image containing 2 channels.

    Returns
    -------
    cropped_nuclei : numpy.array
        Cropped 3D image, only the first channel of the input image.
    cropped_membranes : numpy.array
        Cropped 3D image, only the second channel of the input image.

    """
    
    kernel = morphology.disk(2)

    opened_membranes = cv2.morphologyEx(data[:,1,:,:], cv2.MORPH_OPEN, kernel)

    threshold = filters.threshold_otsu(opened_membranes)
    
    binary_img = opened_membranes > threshold
    
    (z_size, y_size,x_size) = np.shape(opened_membranes)

    left_lim, right_lim = 0, x_size

    for i in range(0,x_size//2,10) :
        if np.max(binary_img[:,:,i:i+10]) == 0 :
            left_lim = i
        
        if np.max(binary_img[:,:,x_size-i-10:x_size-i]) == 0 :
            right_lim = x_size-i
    
    return (data[:,0,:,left_lim:right_lim], data[:,1,:,left_lim:right_lim])



# Retrieving the groove's angle from the image's name
def find_groove_angle_using_fileName(filepath) :
    
    """
    Return the groove's angle (in radians) of the image corresponding to the 
    input filepath

    Parameters
    ----------
    filepath : str
        Name or filepath of the considered image.

    Returns
    -------
    angle : float
        Angle (in radians) of the groove.

    """
    path_decomposition = filepath.split("/")

    name = path_decomposition[-1]

    index = name.rfind('DIL')

    angle_str = name[index+4:index+7]

    if angle_str == '45-' :
        angle = pi/4

    elif angle_str == '90-' :
        angle = pi/2
    
    else :
        angle = pi
    
    return angle


# Contrasts correction for 180° groove images
def correct_contrasts(image):
    
    """
    Artificially correct the contrast differences observed on 180° grooves 
    images, caused by a partial signal loss due to the vertical walls

    Parameters
    ----------
    image : numpy.array (float)
        3D image (1 channel) to treat.

    Returns
    -------
    treated_image : numpy.array
        Treated 3D image.

    """
    
    (Z,Y,X) = np.shape(image)
    
    lim_center = int(round(0.2*X))
    
    side_blurred = filters.gaussian(image[:,:,:50], sigma=1)
    
    side_threshold = filters.threshold_otsu(side_blurred)
    
    binary = side_blurred > side_threshold
    
    z, y, x = np.nonzero(binary)
    
    x_min, x_max = np.min(x), np.max(x)
    
    central_blurred = filters.gaussian(image[:,:,lim_center:-lim_center], sigma=1)
    
    central_threshold = filters.threshold_otsu(central_blurred)
    
    treated_image = np.copy(image)
    
    for i in range(x_min,lim_center,10) :
        right_region = filters.gaussian(image[:,:,i:i+10], sigma=1)
        
        right_threshold = filters.threshold_otsu(right_region)
        
        right_binary = right_region > right_threshold
        
        right_region[right_binary] += (central_threshold - right_threshold)
        
        treated_image[:,:,i:i+10] = right_region
    
    for i in range(x_max,X-lim_center,-10) :
        left_region = filters.gaussian(image[:,:,i-9:i+1], sigma=1)
        
        left_threshold = filters.threshold_otsu(left_region)
        
        left_binary = left_region > left_threshold
        
        left_region[left_binary] += central_threshold - left_threshold
        
        treated_image[:,:,i-9:i+1] = left_region
    
    return treated_image



############################### Axis definition ###############################

def membranes_binarisation(image):
    """
    Return a basic binary image form the grey-level input image 

    Parameters
    ----------
    image : numpy.array (float)
        Membranes channel of the 3D image to treat.

    Returns
    -------
    binary_membranes : numpy.array (binary)
        Treated image, binary image of the membrane channel.

    """
    kernel = morphology.disk(2)

    opened_membranes = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    threshold = filters.threshold_otsu(opened_membranes)

    binary_img = opened_membranes > threshold

    binary_membranes = np.zeros(np.shape(image), dtype=int)
    binary_membranes[binary_img] = 1
    
    return binary_membranes


# Axis definition
def find_intersect_points(image, groove_angle):
    """
    Find the voxels - on each plan along the Y axis - matching the bottom/middle
    of the groove and return their coordinates

    Parameters
    ----------
    image : numpy.array (binary)
        Membranes binary images.
    groove_angle : float
        Size of the groove (pi, pi/2 or pi/4).

    Returns
    -------
    intersect_points : numpy.array
        Coordinates (z,y,x) of the voxels matching the groove's bottom/middle.

    """
    
    (Z, Y, X) = np.shape(image)
                
    intersect_points = []
    
    if groove_angle == pi/2 or groove_angle == pi/4 :
    
        for j in range(Y) :
            
            pixels = np.nonzero(image[:,j,:])
            
            if len(pixels[0]) > 0 :
                z_min = np.min(pixels[0])
                prop_out_pixels = -1
                
                if groove_angle == pi/2 :
                    z, z_max = z_min, z_min + 8
                else :
                    z, z_max = z_min - 10, z_min - 2
                
                # Find x0, the center of the groove (along x axis)
                while z < z_max :
                    liste = []
                    for i in range(X//2-20, X//2+20) :
                        xA, xD = i + X//2, i - X//2
                        
                        zAD = int(z + (X//2) / np.tan(groove_angle/2))
                        
                        a1 = (zAD - z) / (xA - i)
                        b1 = zAD - a1*xA
                    
                        a2 = (zAD - z) / (xD - i)
                        b2 = zAD - a2*xD
                                        
                        z_grid, x_grid = np.ogrid[:Z, :X]
                        
                        stack = np.where(np.logical_and(z_grid > a1*x_grid+b1, z_grid > a2*x_grid+b2), 0, image[z_grid,j,x_grid])
                        
                        all_pixels_l = np.unique(image[:,j,:i], return_counts=True)[1]
                        all_pixels_r = np.unique(image[:,j,i:], return_counts=True)[1]
                        
                        out_pixels_l = np.unique(stack[:,:i], return_counts=True)[1]
                        out_pixels_r = np.unique(stack[:,i:], return_counts=True)[1]
                        
                        if len(all_pixels_l) < 2 or len(out_pixels_l) < 2 :
                            out_over_all_l = 0
                        else :
                            out_over_all_l = out_pixels_l[1] / all_pixels_l[1]
                        
                        if len(all_pixels_r) < 2 or len(out_pixels_r) < 2 :
                            out_over_all_r = 0
                        else :
                            out_over_all_r = out_pixels_r[1] / all_pixels_r[1]
                        
                        
                        if out_over_all_l*out_over_all_r != 0 :
                            liste.append([z, i, abs(out_over_all_l-out_over_all_r), 
                                         (out_pixels_l[1]+out_pixels_r[1])/(all_pixels_l[1]+all_pixels_r[1])])
                    
                
                    if len(liste) > 0 :
                        tab = np.array(liste)
                        p = np.argmin(tab[:,2])
                        
                        z0, x0 = int(tab[p,0]), int(tab[p,1])
                        z = z_max
                        
                        prop_out_pixels = tab[p,3]
                    
                    else :
                        z += 1
                
                
                # Find z0, the bottom of the groove
                while prop_out_pixels > 0.05 :
                    z0 -= 1
                    
                    xA, xD = x0 + X//2-8, x0 - (X//2-8)
                    zAD = int(z0 + (X//2-8) / np.tan(groove_angle/2))
                    
                    a1 = (zAD - z0) / (xA - x0)
                    b1 = zAD - a1*xA
                
                    a2 = (zAD - z0) / (xD - x0)
                    b2 = zAD - a2*xD
                                    
                    z_grid, x_grid = np.ogrid[:Z, :X]
                    
                    stack = np.where(np.logical_and(z_grid > a1*x_grid+b1, z_grid > a2*x_grid+b2), 0, image[z_grid,j,x_grid])
                    
                    all_pixels = np.unique(image[:,j,:], return_counts=True)[1]
                    out_pixels = np.unique(stack[:,:], return_counts=True)[1]
                    
                    if len(out_pixels) > 1 and len(all_pixels) > 1 :
                        prop_out_pixels = out_pixels[1] / all_pixels[1]
                    else :
                        prop_out_pixels = -1
                
                if prop_out_pixels != -1 :
                    intersect_points.append([z0,j,x0])
                
    else :
        for j in range(0,Y,20) :
            
            pixelsZ, pixelsY, pixelsX = np.nonzero(image[:,j:j+20,:])
            
            z0 = np.min(pixelsZ)
            
            x0 = int(round( (np.min(pixelsX) + np.max(pixelsX)) / 2 ))
            
            y0 = int(round(j+(pixelsY[np.argmin(pixelsX)]+pixelsY[np.argmax(pixelsX)] / 2)))
            
            intersect_points.append([z0,y0,x0])
            

    return np.array(intersect_points)




# Use linear regression on intersection points to find the rotation angles 
def find_rotation_angle(xs, ys, nb_fig=1) :
    
    x, y = xs.reshape(-1,1), ys

    # initialisation du modèle
    regression_model = LinearRegression()
    
    # Adapter les données (entraînement du modèle)
    regression_model.fit(x, y)
        
    phi = np.arctan(regression_model.coef_[0])
    
    return phi


def rotation_phi(image, phi) :
    
    (Z,Y,X) = np.shape(image)
    
    rotated = ndimage.rotate(image[:,:,0], math.degrees(phi), reshape=True)
    
    (new_Z, new_Y) = np.shape(rotated)
    
    transformed_image = np.zeros((new_Z, new_Y, X), dtype=int)
    
    transformed_image[:,:,0] = rotated
    
    for k in range(1,X) :
        transformed_image[:,:,k] = ndimage.rotate(image[:,:,k], math.degrees(phi), reshape=True)
    
    return transformed_image


# Use linear regression on intersection points to find x0 and z0
def find_axis_origins(intersect_points):
    zs, ys, xs = intersect_points[:,0], intersect_points[:,1], intersect_points[:,2]
    
    ## Find z0
    x, y = ys.reshape(-1,1), zs
    regression_model = LinearRegression()
    regression_model.fit(x, y)
    y_predicted = regression_model.predict(x)
    
    z0 = int(round((y_predicted[0]+y_predicted[-1])/2))
    
    ## Find x0
    y = xs
    regression_model = LinearRegression()
    regression_model.fit(x, y)
    y_predicted = regression_model.predict(x)
    
    x0 = int(round((y_predicted[0]+y_predicted[-1])/2))
    
    return (z0, x0)

    


############################# Nuclei segmentation #############################

def crop_nucleus(labelled_nuclei, label):
    """
    Crop the image around the nuclei of the given label, return the cropped 
    image and the position in the original image

    Parameters
    ----------
    labelled_nuclei : numpy.array (int)
        Labelled connected components image.
    label : int
        Label of the object to crop / isolate from the image.

    Returns
    -------
    tuple (numpy.array, tuple of int)
        Cropped image and the coordinates of the voxel (0,0,0) of the cropped 
        image in the global image.

    """
        
    (z_size, y_size,x_size) = np.shape(labelled_nuclei)
    
    left_lim, right_lim = 0, x_size
    
    binary_img = labelled_nuclei == label
    
    (x, y, z) = np.nonzero(binary_img)
    
    down_lim, up_lim = np.min(x), np.max(x)
    
    front_lim, back_lim = np.min(y), np.max(y)
    
    left_lim, right_lim = np.min(z), np.max(z)
    
    Z, Y, X = up_lim - down_lim + 10, back_lim - front_lim + 10, right_lim - left_lim + 10
    
    cropped_img = np.zeros((Z, Y, X), dtype=int)
    
    cropped_img[5:Z-5, 5:Y-5, 5:X-5] = binary_img[down_lim:up_lim,front_lim:back_lim,left_lim:right_lim]
    
    return (cropped_img, (down_lim, front_lim, left_lim))




def local_binary_watershed(labelled_img, nb_nuclei):
    """
    Apply the binary watershed algorithm on the given image

    Parameters
    ----------
    labelled_img : numpy.array (int)
        Small image (containing only one labelled object) to apply the BW 
        algorithm on.
    nb_nuclei : int
        Number of regions to subdivide the labelled object in.

    Returns
    -------
    labels : numpy.array (int)
        Final labelled image.

    """
    
    (img, nb_components) = ndimage.label(labelled_img)
    
    distances = ndimage.distance_transform_edt(img)
    
    n = 9
    
    coords = feature.peak_local_max(distances, min_distance=n, num_peaks=nb_nuclei)
    
    nb_peaks = len(coords)
    
    while nb_peaks < nb_nuclei and n > 4:
        n -= 1
        
        coords = feature.peak_local_max(distances, min_distance=n, num_peaks=nb_nuclei)
        
        nb_peaks = len(coords)
        
    mask = np.zeros(distances.shape, dtype=bool)
    
    mask[tuple(coords.T)] = True
    
    markers, _ = ndimage.label(mask)
    
    labels = segmentation.watershed(-distances, markers, mask=img)
    
    return labels





def connexe_component_division(labelled_frame, volume, v_min):
    """
    Find the best number of regions to subdivide a connected component in.

    Parameters
    ----------
    labelled_frame : numpy.array
        Small image containing only the region (nuclei or group of nuclei) of 
        interest.
    volume : int
        Volume of the region of interest.
    v_min : int
        Minimum volume acceptable for a nuclei.

    Returns
    -------
    n : int
        Number of objects alowing to get the best mean convexity after the 
        application of the BW algorithm.

    """
    n = volume//v_min
    
    if n == 0 :
        n = 1
    
    else :
        convexities = np.zeros((n,n))
        
        convexities[0,0] = measure.regionprops(labelled_frame)[0].solidity
        if n > 1 :
            for i in range(2, n+1):
                treated_frame = local_binary_watershed(labelled_frame, nb_nuclei=i)
                # if i < 6 : 
                #     viewer.add_labels(treated_frame)
                
                volumes = np.unique(treated_frame, return_counts=True)[1]
                k = len(volumes)
                
                props = measure.regionprops(treated_frame)
                if np.min(volumes) >= v_min :
                    for j in range(k-1):
                        convexities[i-1,j] = props[j].solidity
                else :
                    convexities[i-1,0] = 0.01
                        
            convexities_means = np.mean(convexities, axis=1, where=convexities>0)
            # print(convexities)
            # print(convexities_means)
            
            n = np.argmax(convexities_means) + 1
        
    # print(convexities[n-1])
    return n



def reassigne_label(frame, n, new_label) :
    """
    Reassigne labels to the segmented nuclei in the final whole image

    Parameters
    ----------
    frame : numpy.array
        Small image containing only the region (nuclei or group of nuclei) of 
        interest.
    n : int
        Number of objects alowing to get the best mean convexity after one 
        application of the BW algorithm.
    new_label : int
        Label to assign to the first newly segmented nuclei of this frame.

    Returns
    -------
    resulting_frame : numpy.array (int)
        Segmented nuclei frame, with the right (new) labels.
    new_label : int
        Label to assign to the first newly segmented nuclei of the next frame
        to process.

    """
    if n == 1 :
        resulting_frame = np.copy(frame)
        
        resulting_frame[resulting_frame==1] = new_label
        # print("1", new_label)

        new_label += 1
        
    elif n > 1 :
        treated_frame = local_binary_watershed(frame, nb_nuclei=n)
        
        resulting_frame = np.copy(treated_frame)
        
        labels, volumes = np.unique(treated_frame, return_counts=True)
        # view_labelled_napari(treated_frame, "n="+str(n))
        # print(volumes)
        for k in range(1, min(n+1, len(labels))):
            if volumes[k] > 3000 :
                big_nucleus = measure.label(treated_frame == k)
                n1 = connexe_component_division(big_nucleus,volumes[k], 1200)
                treated_nucleus = local_binary_watershed(big_nucleus, nb_nuclei=n1)
                # print("n1:",n1)
                
                for p in range(1, min(n1+1, len(np.unique(treated_nucleus)))):
                    resulting_frame[treated_nucleus==p] = new_label
                    # print("2", new_label)
                    new_label += 1
                    
            else :
                resulting_frame[treated_frame==k] = new_label
                # print("3", new_label)
                # print("New label attributed for old label", p,":", new_label)
                new_label += 1
    
    return resulting_frame, new_label


# Main function of nuclei segmentation
def otsu_binaryWatershed(img_nuclei) :
    """
    Global nuclei segmentation algorithm

    Parameters
    ----------
    img_nuclei : numpy.array (float)
        Pre-treated (rescaling, contrasts correction for 180° images) image, 
        nuclei channel.

    Returns
    -------
    resulting_img : numpy.array (int)
        Final labelled segmented image.

    """

    blurred = filters.gaussian(img_nuclei, sigma=1)

    thresh = filters.threshold_otsu(blurred)

    binary = blurred > thresh
    
    resulting_img, new_label = np.zeros(np.shape(img_nuclei), dtype=int), 1
    
    (labelled_img, nb_components) = ndimage.label(binary)
    
    labels, volumes = np.unique(labelled_img, return_counts=True)
        
    v_min1, v_min2 = 500, 600 
    
    for p in labels[1:] :
                               
        #Apply Binary Watershed if needed
        if volumes[p] < v_min1 :
            resulting_img[labelled_img==p] = 0
        
        else :
            (frame, indexes) = crop_nucleus(labelled_img, p)
            
            n = connexe_component_division(frame, volumes[p], v_min2)
            
            (treated_frame, new_label) = reassigne_label(frame, n, new_label)                        
    
            (Z,Y,X) = np.shape(treated_frame)
                
            resulting_img[indexes[0]:indexes[0]+Z-10, indexes[1]:indexes[1]+Y-10, indexes[2]:indexes[2]+X-10] += treated_frame[5:Z-5, 5:Y-5, 5:X-5]
    
    
    return resulting_img



################################ Image analysis ###############################

def distance(coord1, coord2):
    """
    

    Parameters
    ----------
    coord1 : numpy.array
        Coordinates of the first voxel.
    coord2 : numpy.array
        Coordinates of the second voxel.

    Returns
    -------
    distance : float
        Distance between the two voxels (computed in an orthonormal 2D or 3D
        reference frame).

    """
    if len(coord1) != len(coord2) :
        print("ERROR: coord1 and coord2 must be of the same length")
    
    else :
        s = 0
        for i in range(len(coord1)) :
            s += (coord1[i]-coord2[i])**2
        
        return math.sqrt(s)
            

def boundaries_detection(labelled_img):
    """
    

    Parameters
    ----------
    labelled_img : numpy.array (int)
        Labelled segmented nuclei image.

    Returns
    -------
    numpy.array (int)
        Labelled boundaries between the nuclei.

    """
        
    boundaries = segmentation.find_boundaries(labelled_img, mode='outer')
    # viewer.add_labels(boundaries)
    
    binary_img = labelled_img > 0
    
    external_boundaries = segmentation.find_boundaries(binary_img, mode='outer')

    return boundaries * (1-external_boundaries) * labelled_img



def detect_groups(labelled_img):
    """
    Compute the number and size of the groups of touching nuclei in the 
    segmented image

    Parameters
    ----------
    labelled_img : numpy.array (int)
        .

    Returns
    -------
    group_sizes : int list
        Number of groups (list element) of nuclei in relation to their 
        size (list index).

    """
    
    connexe_comp, nb_components = ndimage.label(labelled_img)
    # viewer.add_labels(connexe_comp)
        
    group_sizes = [0]
    
    for i in range(1, nb_components) :
        group_size = len(np.unique(labelled_img[connexe_comp==i]))
        
        group_max = len(group_sizes)
        
        if group_max < group_size :
            group_sizes.extend([0 for j in range(group_size-group_max)])    
        
        group_sizes[group_size-1] += 1
    
    return group_sizes



def count_contacts(labelled_img) :
    """
    Count the number of touching pairs of nuclei

    Parameters
    ----------
    labelled_img : numpy.array
        Labelled segmented nuclei image.

    Returns
    -------
    int
        Number of touching pairs of nuclei.

    """
    
    boundaries = boundaries_detection(labelled_img)
    
    nb_nuclei = np.max(boundaries)
    
    labelled_boundaries, nb_components = ndimage.label(boundaries)
    
    neighbors_matrix = np.zeros((nb_nuclei+1,nb_nuclei+1), dtype=int)
    
    props = measure.regionprops(labelled_boundaries)
    
    for i in range(1, nb_components) :
        
        (z_min, y_min, x_min, z_max, y_max, x_max) = props[i-1].bbox
        
        frame = boundaries[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
        labelled_frame = labelled_boundaries[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
        
        neighbors = np.unique(frame[labelled_frame==i])
        
        if len(neighbors) == 2 :
            
            neighbors_matrix[neighbors[0], neighbors[1]] = 1
            neighbors_matrix[neighbors[1], neighbors[0]] = 1
        
        else :
            for j, nuc in enumerate(neighbors) :
                for k in range(j+1, len(neighbors)) :
                    img_neighbors = frame[np.logical_or(frame==nuc, frame==neighbors[k])]
                    labels, nb_comps = ndimage.label(img_neighbors)
                    
                    if nb_comps == 1 :
                        neighbors_matrix[nuc, k] = 1
                        neighbors_matrix[k, nuc] = 1
        
    
    return (np.sum(neighbors_matrix) // 2)
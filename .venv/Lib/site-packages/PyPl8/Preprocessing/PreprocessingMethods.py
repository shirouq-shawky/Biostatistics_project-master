# Load Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageOps
import glob
import os
from shutil import copy2

from pathlib import Path
from scipy import ndimage as ndi
import skimage
from skimage import feature, filters, data, color, io
from skimage.color import label2rgb

from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, remove_small_objects, rectangle
from skimage.morphology.convex_hull import convex_hull_image
from scipy.signal import find_peaks
from IPython.display import clear_output
from math import floor
import collections

import sys
sys.path.append('./')

#import argparse
import cv2

# Functions
def ListFiles(folderpath):
    '''
    Make a list of all the files in given folder that is in the current working directory
    '''
    path = Path(folderpath)
    files = os.listdir(path)
    return files

def LoadImage(file, sourcefolderpath, rotate = False):
    '''
    Change Working Directory to image folder and read image file using skimage
    '''
    wd = os.getcwd()
    path = Path(sourcefolderpath)   
    os.chdir(path)
    if rotate:
        image = ndi.rotate(skimage.io.imread(file),180)
    else:
        image = skimage.io.imread(file)
    os.chdir(Path(wd))
    return image

def DisplayTiles(tile_list, dimensions=(8,12)):
    '''
    Display tiles inline in a grid format 
    '''
    #clear_output(wait=True)
    num_rows = dimensions[0]
    num_cols = dimensions[1]
    fig, axs = plt.subplots(num_rows, num_cols,figsize=(num_cols,num_rows))
    count = 0
    for j in range(0,num_rows):
        for jj in range(0,num_cols):
            axs[j,jj].imshow(tile_list[count])
            axs[j,jj].axis('off')
            count = count + 1
    plt.show();
    
def MergeTiles(image, tiles, corners):
    '''
    Input: tiles-- list of rectangular regions of interest selected from full image,
           corners-- list of top left corners of giving location of tiles in the full image
           image-- original image from which tiles were cropped
           
    Output: img -- image with tiles replacing locations in original image
    
    This function is used to reconstruct the full segmented image for visualization
    '''
    s = np.shape(tiles[0])[0]
    if len(np.shape(tiles[0]))==3:
        gray_tiles = [skimage.color.rgb2gray(t) for t in tiles]
    else:
        gray_tiles = tiles.copy()
        
    if len(np.shape(image))==3:
        img = skimage.color.rgb2gray(image)
    elif len(np.shape(image))==2:
        img = image
    else:
        print("Input image must be grayscale or RGB.")
    
    for c,t in zip(corners,gray_tiles):
        r1,c1 = c
        img[r1:r1+s,c1:c1+s] = t
    
    return img

def find_nearest(array, value):
    '''
    Find the element in the reference array which most closely matches the input value
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def AutoAdjust(image, corners, s = 200, array_dimensions = (8,12)):
    '''
    input: image-- original plate image
           corners -- initial guess at corner locations of tiles
           s -- side length of tiles
           array_dimensions
          
    output: 
           new_corners: list of top left corners of tiles 
           
    Uses corners as an initial guess at creating tiles from image, then 
    adjusts such that each tile is centered on the centroid of the object contained in the tile.
    The corners of these adjusted tile locations are returned. 
    '''
    if len(np.shape(image))==3:
        img = skimage.color.rgb2gray(image)
    elif len(np.shape(image))==2:
        img = image
    else:
        print("Input image must be grayscale or RGB.")
        
    tiles = MakeTiles(img,corners)
    n = round(s/2)
    
    columnshifts = np.zeros(array_dimensions[1])
    rowpositions =  np.zeros(array_dimensions[0])
    tilestoadjust=[]
    new_corners = [None]*len(tiles)
    
    j = 0
    for r in range(0,array_dimensions[0]): # loop over rows
        row_position=[]
        for c in range(0,array_dimensions[1]): # loop over columns
            t = tiles[j]
            r1,c1 = corners[j]
            m = remove_small_objects(t>.6, min_size = 0.05*s**2)
            labels = label(m)
            if np.max(labels):
                props = regionprops(labels)
                minr, minc, maxr, maxc = props[0].bbox
                if (maxr-minr)/(maxc-minc) < .5 or (maxc-minc)/(maxr-minr)<.5:
                    if len(props) > 1:
                        rs = [np.abs(p.perimeter/2/np.pi-np.sqrt(p.area/np.pi)) for p in props]
                        idx = np.argmin(rs)
                        shift_x = props[idx].centroid[0] - n
                        shift_y = props[idx].centroid[1]- n            
                        row_position.append(r1 + int(shift_x))
                        columnshifts[j%array_dimensions[1]] = columnshifts[j%array_dimensions[1]]+shift_y
                    else: 
                        shift_x = 0
                        shift_y = 0
                        tilestoadjust.append(j)
                else:   
                    if len(props) > 1:
                        rs = [np.abs(p.perimeter/2/np.pi-np.sqrt(p.area/np.pi)) for p in props]
                        idx = np.argmin(rs)
                    else:
                        idx = 0
                    shift_x = props[idx].centroid[0] - n
                    shift_y = props[idx].centroid[1]- n
                    
                    row_position.append(r1 + int(shift_x))
                    columnshifts[j%array_dimensions[1]] = columnshifts[j%array_dimensions[1]]+shift_y

            else:
                shift_x = 0
                shift_y = 0
                tilestoadjust.append(j)
            new_corners[j] = (r1 + int(shift_x),c1+int(shift_y))        
            j = j + 1
        
        if row_position:
            rowpositions[r] = int(np.mean(row_position))
        else:
            rowpositions[r] = float('nan')

    colshifts = [int(c/array_dimensions[0]) for c in columnshifts]
    try:
        if np.isnan(rowpositions).any():
            indices = np.argwhere(np.isnan(rowpositions))[0]
            for idx in indices:
                if idx == 0:
                    rowpositions[0] = rowpositions[1] - s
                elif idx == len(rowpositions)-1:
                    rowpositions[len(rowpositions)-1] = rowpositions[len(rowpositions)-2] + s
                else:
                    rowpositions[idx] = int((rowpositions[idx-1]+rowpositions[idx+1])/2)
    except:
        print('Autoadjust failed in preprocessing because colony growth is too faint. Try grid crop without auto adjust.')
    # adjust placement of tiles that appeared empty as well based on average adjustment across strong growers
    
    j = 0
    for c in new_corners:
        r1, c1 = c
        if j in tilestoadjust:
            r1 = rowpositions[int(np.floor(j/array_dimensions[1]))]
            c1 = c1+colshifts[j%array_dimensions[1]]            
        new_corners[j] = (int(r1),int(c1))
        j = j+1    
    return new_corners
        
     

def MakeTiles(image, corners, s = 200):
    '''
    input: image-- original plate image
           corners-- list of top left corners of tile locations
           s -- desired sidelength for tiles
    output: tiles-- list of square regions of interest cropped with a square of side length s with
            top left corner for each element of corners           
    '''    
    if len(np.shape(image))==3:
        img = skimage.color.rgb2gray(image)
    elif len(np.shape(image))==2:
        img = image
    else:
        print("Input image must be grayscale or RGB.")
    tiles = [None]*len(corners)
    for i, c in enumerate(corners):
        r1,c1 = c    
        temp = img[r1:r1+s,c1:c1+s]
        tiles[i] = temp.copy()
    return tiles

# Cropping Options
def AutoCrop(file, sourcefolderpath, rotate = False, 
              s = 200, array_dimensions = (8,12), 
              adjust=True, display = False):
    '''
    Input: RGB or grayscale image of rectangular arrayed plate.
           parameter: plate height in pixels
           s side length of desired tiles in pixels
           array_dimensions number of rows and columns in pin arrangement
           Display True or False indicating whether to show resulting tiles on screen
           Adjust True or False indicating whether to adjust tile location with 
                       the AutoAjust function defined above or just use the initial guess grid
    Returns: 
            tiles --Ordered list of subsections of the image containing a single colony, 
                or blanks if there are empty spots in the array 
            corners -- list of top left corners corresponding to location of these tiles
    '''
    image = LoadImage(file, sourcefolderpath, rotate = rotate)
    
    # Convert to gray_scale if necessary. Return a message if image is in unexpected format.
    if len(np.shape(image))==3:
        img = skimage.color.rgb2gray(image)
    elif len(np.shape(image))==2:
        img = image
    else:
        print("Input image must be grayscale or RGB.")
        
    # Crop to plate by identifying abrupt change from black background to plate edge.    
    x = np.sum(img > 0.5, axis = 0)
    thresholdedx = [c[0] for c in np.argwhere(x >1700)] #rough estimate of agar height
    ddx = np.diff(thresholdedx)
    idx = np.argmax(ddx)
    c1 = thresholdedx[idx]; c2 = thresholdedx[idx+1]

    # -- Identify top and bottom
    bw = img>.5
    bw = remove_small_objects(bw, min_size =0.025*s**2)
    mask = convex_hull_image(bw)

    y = np.sum(mask, axis = 1)
    dd = np.diff(y>0)
    r1 = np.argwhere(dd>0)[0][0]
    r2 = np.argwhere(dd>0)[1][0]
    
    # -- Set initial guess for grid estimating where pin points were
    bufferr = 250 # buffer away from plate edges
    bufferc = 175
    rps = np.linspace(r1+bufferr, r2-bufferr, array_dimensions[0])
    cps = np.linspace(c1+bufferc, c2-bufferr, array_dimensions[1])
    corners = [None]*len(rps)*len(cps)
    j = 0
    for r in rps:
        for c in cps:
            corners[j] = (int(r-s/2),int(c-s/2))
            j = j+ 1
    # -- Adjust to center each patch on a tile
    if adjust:
        new_corners = AutoAdjust(img, corners, s = s, array_dimensions = array_dimensions)
        corners = AutoAdjust(img, new_corners, s = s, array_dimensions = array_dimensions)
    # -- Crop tiles    
    tiles = MakeTiles(img, corners, s = s)        
    return tiles, corners
             
    
def GridCrop(file, sourcefolderpath, rotate = False, 
              A1_location = (500,500), s = 200, array_dimensions = (8,12), 
              adjust=True, display = False):
    '''
    input: name of image file
           expected dimensions of array of colonies on plate
           A1_location: estimate of center of pin location A1
           expected tile size in pixels           
           Adjust -- True or False indicating whether to automatically adjust 
           tile locations to be centered on bright patches
    returns: tiles-- list of square regions of interest defined 
            corners -- list of top left corners of tiles
    
    '''
    image = LoadImage(file, sourcefolderpath, rotate = rotate)
    
    # Convert to gray_scale if necessary. Return a message if image is in unexpected format.
    if len(np.shape(image))==3:
        img = skimage.color.rgb2gray(image)
    elif len(np.shape(image))==2:
        img = image
    else:
        print("Input image must be grayscale or RGB.")
        
    # -- Initiate grid locations based on A1    
    r1 = A1_location[0]
    r2 = r1 + (array_dimensions[0]-1/2)*s
    c1 = A1_location[1]
    c2 = c1 + (array_dimensions[1]- 1/2)*s*1.02

    rps = np.linspace(r1,r2,8)
    cps = np.linspace(c1,c2,12)
    
    corners = [None]*len(rps)*len(cps)
    j = 0
    for r in rps:
        for c in cps:
            corners[j] = (int(r-s/2),int(c-s/2))
            j = j+ 1
    
    # -- Adjust to center each patch on a tile
    if adjust:
        new_corners = AutoAdjust(img, corners, s = s, array_dimensions = array_dimensions)
        corners = AutoAdjust(img, new_corners, s = s, array_dimensions = array_dimensions)
    # -- Crop tiles    
    tiles = MakeTiles(img, corners, s = s)        
    return tiles, corners
       
refPt = []
output = None
def ClickCrop(file, sourcefolder, s=200, array_dimensions = (8,12), rotate = False, adjust = True):
    '''
    Click on center of patch in the A1 position to crop image into tiles based on this 
    reference point.
    input: name of image file
           rotate = True or False indicating whether image needs to be 
           rotated 180* when loaded
           expected tile side length in pixels
           expected dimensions of array of colonies on plate
           Adjust -- True or False indicating whether to automatically adjust 
           tile locations to be centered on bright patches
    returns: tiles-- list of square regions of interest defined 
            corners -- list of top left corners of tiles
    
    '''
    # initialize the list of reference points and boolean indicating
    # whether cropping is being performed or not   
    global refPt, output
    # Load and resize image
    if rotate:
        im = ImageOps.mirror(ImageOps.flip(Image.open(os.path.join(sourcefolder,file))))
    else:
        im = Image.open(os.path.join(sourcefolder,file))
    height,width = im.size
    image = np.array(im.resize((int(height/4),int(width/4))))
    
    #Define click and crop event on image
    def click_and_crop(event, x, y, flags, param):
        # grab references to the global variables
        global refPt, output
        # if the left mouse button was clicked, record the position
        # (x, y) coordinates 
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt = [(x, y)]
        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # draw a rectangle around the region of interest
            n = round(100/4)
            A1_location = (refPt[0][1],refPt[0][0])
            r1 = A1_location[0]
            r2 = r1 + (array_dimensions[0]-1/2)*2*n
            c1 = A1_location[1]
            c2 = c1 + (array_dimensions[1]- 1/2)*2*n*1.02

            rps = np.linspace(r1,r2,8)
            cps = np.linspace(c1,c2,12)
            
            corners = [None]*len(rps)*len(cps)
            cv2.rectangle(image, (int(c1-n),int(r1-n)), (int(c1+n),int(r1+n)), (0, 255, 0), 2)   
            j = 0
            for r in rps:
                for c in cps:
                    corners[j] = (int((r-n)*4),int((c-n)*4))
                    cv2.rectangle(image, (int(c-n),int(r-n)), (int(c+n),int(r+n)), (255, 0, 0), 2) 
                    j = j+1
            output = corners.copy()
            cv2.imshow("image", image)
               
    # clone downsized image, and setup the mouse callback function
    clone = image.copy()
    big_clone = np.array(im.copy())
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    
    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("n"):
            image = clone.copy()
            output = None
        # if the 'c' key is pressed, break from the loop
        elif key == ord("y"):
            break
       
    # close all open windows
    cv2.destroyAllWindows()
    corners = output
    if adjust:
        new_corners = AutoAdjust(big_clone, corners, s = s, array_dimensions = array_dimensions)
        corners = AutoAdjust(big_clone, new_corners, s = s, array_dimensions = array_dimensions)
    # -- Crop tiles    
    tiles = MakeTiles(big_clone, corners, s = s)        
    return tiles, corners

def GridDisplay(image):
    fig, axs = plt.subplots(1,1,figsize=(12,16))
    axs.imshow(image);
    axs.grid(which = 'both', linestyle='-', color='r')
    plt.show();
    return None

def EnhancedGridDisplay(image, corner, s, array_dimensions):
    fig, axs = plt.subplots(1,1,figsize=(12,16))
    axs.imshow(image);
    axs.grid(which = 'both', linestyle='-', color='r')
    r1 = corner[0]
    r2 = r1 + (array_dimensions[0]-1/2)*s
    c1 = corner[1]
    c2 = c1 + (array_dimensions[1]- 1/2)*s*1.02

    rps = np.linspace(r1,r2,8)
    cps = np.linspace(c1,c2,12)
    
    for r in rps:
        for c in cps:
            r1 = int(r-s/2)
            c1 = int(c-s/2)
            rect = patches.Rectangle((c1, r1), s, s, linewidth=2, edgecolor='b', facecolor='none')
            axs.add_patch(rect)
    plt.show();
    return None

def AutoCropTest(image, s = 200, array_dimensions = (8,12),adjust=True, display = False):
    '''
    Cropping adjusted for test images included in package (ie image is already loaded as an array)
    Input: RGB or grayscale image of rectangular arrayed plate.
           parameter: plate height in pixels
           s side length of desired tiles in pixels
           array_dimensions number of rows and columns in pin arrangement
           Display True or False indicating whether to show resulting tiles on screen
           Adjust True or False indicating whether to adjust tile location with 
                       the AutoAjust function defined above or just use the initial guess grid
    Returns: 
            tiles --Ordered list of subsections of the image containing a single colony, 
                or blanks if there are empty spots in the array 
            corners -- list of top left corners corresponding to location of these tiles
    '''
    
    # Convert to gray_scale if necessary. Return a message if image is in unexpected format.
    if len(np.shape(image))==3:
        img = skimage.color.rgb2gray(image)
    elif len(np.shape(image))==2:
        img = image
    else:
        print("Input image must be grayscale or RGB.")
        
    # Crop to plate by identifying abrupt change from black background to plate edge.    
    x = np.sum(img > 0.5, axis = 0)
    thresholdedx = [c[0] for c in np.argwhere(x >1700)] #rough estimate of agar height
    ddx = np.diff(thresholdedx)
    idx = np.argmax(ddx)
    c1 = thresholdedx[idx]; c2 = thresholdedx[idx+1]

    # -- Identify top and bottom
    bw = img>.5
    bw = remove_small_objects(bw, min_size =1000)
    mask = convex_hull_image(bw)

    y = np.sum(mask, axis = 1)
    dd = np.diff(y>0)
    r1 = np.argwhere(dd>0)[0][0]
    r2 = np.argwhere(dd>0)[1][0]
    
    # -- Set initial guess for grid estimating where pin points were
    bufferr = 250 # buffer away from plate edges
    bufferc = 175
    rps = np.linspace(r1+bufferr, r2-bufferr, array_dimensions[0])
    cps = np.linspace(c1+bufferc, c2-bufferc, array_dimensions[1])
    corners = [None]*len(rps)*len(cps)
    j = 0
    for r in rps:
        for c in cps:
            corners[j] = (int(r-s/2),int(c-s/2))
            j = j+ 1
    # -- Adjust to center each patch on a tile
    if adjust:
        new_corners = AutoAdjust(img, corners, s = s, array_dimensions = array_dimensions)
        corners = AutoAdjust(img, new_corners, s = s, array_dimensions = array_dimensions)
    # -- Crop tiles    
    tiles = MakeTiles(img, corners, s = s)        
    return tiles, corners
             
    
def GridCropTest(image, A1_location = (500,500), s = 200, array_dimensions = (8,12), 
              adjust=True, display = False):
    '''
    Cropping adjusted for test images included in package (ie image is already loaded as an array)
    input: image array
           expected dimensions of array of colonies on plate
           A1_location: estimate of center of pin location A1
           expected tile size in pixels           
           Adjust -- True or False indicating whether to automatically adjust 
           tile locations to be centered on bright patches
    returns: tiles-- list of square regions of interest defined 
            corners -- list of top left corners of tiles
    
    '''
   
    # Convert to gray_scale if necessary. Return a message if image is in unexpected format.
    if len(np.shape(image))==3:
        img = skimage.color.rgb2gray(image)
    elif len(np.shape(image))==2:
        img = image
    else:
        print("Input image must be grayscale or RGB.")
        
    # -- Initiate grid locations based on A1    
    r1 = A1_location[0]
    r2 = r1 + (array_dimensions[0]-1/2)*s
    c1 = A1_location[1]
    c2 = c1 + (array_dimensions[1]- 1/2)*s*1.02

    rps = np.linspace(r1,r2,array_dimensions[0])
    cps = np.linspace(c1,c2,array_dimensions[1])
    
    corners = [None]*len(rps)*len(cps)
    j = 0
    for r in rps:
        for c in cps:
            corners[j] = (int(r-s/2),int(c-s/2))
            j = j+ 1
    
    # -- Adjust to center each patch on a tile
    if adjust:
        new_corners = AutoAdjust(img, corners, s = s, array_dimensions = array_dimensions)
        corners = AutoAdjust(img, new_corners, s = s, array_dimensions = array_dimensions)
    # -- Crop tiles    
    tiles = MakeTiles(img, corners, s = s)        
    return tiles, corners   
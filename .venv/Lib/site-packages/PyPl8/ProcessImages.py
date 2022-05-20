# Load Packages
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from PIL import Image
from skimage import feature, filters, data, color, io
import skimage
from skimage.transform import hough_circle
from skimage.feature import peak_local_max
from skimage.draw import disk
from scipy.signal import find_peaks
from IPython.display import clear_output
import threading

import pkg_resources
import sys
sys.path.append('./')
from skimage.filters import sobel
from scipy.signal import find_peaks
from pathlib import Path
import glob
import os
import shutil
import pandas as pd
from skimage.exposure import histogram

from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from shutil import copy2
import cv2

from .Preprocessing import PreprocessingMethods as PP

# -- Functions for Segmentation
def CalculateBackground(tiles):
    '''
    input: list of color or grayscale images each containing a single patch
    output: scalar estimate of the intensity of the background agar based on the top 1/10th of each image
    '''
    if len(np.shape(tiles[0]))==3:
        gray_tiles = [skimage.color.rgb2gray(t) for t in tiles]
    else:
        gray_tiles = tiles.copy()
    # Compute average background value of agar
    background_list = [np.median(t[0:int(np.shape(t)[0]/10),0:-5]) for t in gray_tiles]
    background = np.mean(background_list)
    return background

def SegmentTile(tile, background, pin_size=25, threshold_method='otsu'):
    '''
    input: required - color or grayscale tile containing a single patch
                    - scalar estimate of background intensity of agar
           optional
                    - radius of the initial pin (minimum colony size) in number of pixels. Default value is 25. Use pin_size = 0 to exclude circle detection from the process.
                    - method for intensity thresholding. Default value is 'otsu'. Currently this is the only option
    output:
            mask with 1's indicating a pixel is part of the colony and 0's indicating a pixel is part of the background
    '''
    # -- convert to grayscale
    if len(np.shape(tile))==3:
        t = skimage.color.rgb2gray(tile)
    else:
        t = tile.copy()
    # -- initial intensity threshold    
    ss = np.shape(t)[0]
    if threshold_method == 'otsu' or threshold_method == 'Otsu':
        buffer = int(ss/8)
        thresh = threshold_otsu(t[buffer:-buffer,buffer:-buffer])     
    else:
        print('Threshold method not recognized')
    # -- binarize and fill
    bw = closing(t > thresh, square(3))
    bw = ndi.binary_fill_holes(bw)    
    label_image = label(bw)
    label_objects, nb_labels = ndi.label(label_image>0)
    # Select largest object
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes > .05*(ss**2)
    mask_sizes[0] = 0
    colony_mask_temp = mask_sizes[label_objects]
    # remove artifacts connected to image border
    colony_mask = clear_border(colony_mask_temp)
    
    # -- For large patches, use threshold results. To leave out circle fitting for everything, use pin_size = 0
    # -- For small patches fit circle to tile with min radius of pin_size, max of pin_size + 10
    if pin_size > 0 and (np.sum(colony_mask) <= np.pi*(pin_size+10)**2 or thresh < background + 0.035): 
        # Circular Hough Transform
        buffer = int(0.175*ss)
        edges = 1*(t>np.mean(t[buffer:-buffer,buffer:-buffer])+.01)
        edges_center = edges[buffer:-buffer,buffer:-buffer]
        hough_radii = np.arange(pin_size,pin_size+10, 1)
        hough_res = hough_circle(edges_center, hough_radii)
        centers = [None]*len(hough_radii) 
        accums = [None]*len(hough_radii) 
        radii = [None]*len(hough_radii)
        for i, (radius, h) in enumerate(zip(hough_radii, hough_res)):
            # For each radius, extract one circle
            peaks = peak_local_max(h, num_peaks=1, min_distance=int(ss/8))
            centers[i] = peaks.tolist()[0]
            accums[i] = h[peaks[:, 0],peaks[:,1]].tolist()[0]
            radii[i] = radius
            
        # Consider Likelihood of candidate circles.
        if np.max(accums) <= 0.42: # change tile to blank if circles all less than 42%
            colony_mask = np.zeros_like(t)>0
        else: # Pick circle with highest likelihood that is less than 92% for mask 
            # (higher indicates that the circle is entirely contained in patch, not on the edge)
            accums = accums*np.array([x < 0.92 for x in accums])
            idx = np.argmax(accums)
            center_x, center_y = centers[idx]
            radius = radii[idx]
            # Build mask
            cx, cy = disk((center_x, center_y), radius)
            colony_mask_center = np.zeros_like(edges_center)
            colony_mask = np.zeros_like(edges)
            try:
                colony_mask_center[cx,cy] = 1
                colony_mask[buffer:-buffer,buffer:-buffer] = colony_mask_center
            except:
                pass
            # Boolean-ize it
            colony_mask = colony_mask>0
            # Combine with thresholding result if the colony is sufficiently brighter than background
            if thresh> background + 0.035: 
                label_objects, nb_labels = ndi.label(1*((label_image+colony_mask)>0))
                # Select largest object
                sizes = np.bincount(label_objects.ravel())
                mask_sizes = sizes > .05*(ss**2)
                mask_sizes[0] = 0
                colony_mask_temp = mask_sizes[label_objects]
                # remove artifacts connected to image border
                colony_mask = clear_border(colony_mask_temp)
    return colony_mask

def BuildSegmentedImage(image,colony_masks,corners):
    '''
    input: - original image that was segmented
           - list of masks for each tile from the image
           - list of top left corners indicating location of tile in the original image
    return: black and white version of image with segmented colonies colored in    
    '''
    if len(np.shape(image))==3:
        img = skimage.color.rgb2gray(image)
    else:
        img = image.copy()
    
    label_image = label(1*(PP.MergeTiles(np.zeros_like(img),colony_masks, corners)>0))
    image_label_overlay = label2rgb(label_image, image=img, bg_label=0)   
    return image_label_overlay

# -- Functions for Quantitative Feature Extraction

def SizeFeatures(tile,mask):
    '''
    input: tile containing a single colony of interest and corresponding mask indicating segmentation
    returns: list containing area of colony, average pixel intensity of colony, sum of pixel intensities in colony, background intensity, perimeter of colony
    '''
    if len(np.shape(tile))==3:
        gray_tile = skimage.color.rgb2gray(tile)
    else:
        gray_tile = tile.copy()
    clean_tile = tile*mask
    area = np.sum(1*mask)
    pixel_sum = np.sum(clean_tile)
    if area > 0:
        avg_int = pixel_sum/area
    else:
        avg_int = 0
    perimeter = skimage.measure.perimeter(mask)
    background = np.sum(tile*(1-1*mask))/np.sum((1-1*mask))
    return [area, avg_int, pixel_sum, background, perimeter]

def TextureFeatures(tile,mask):
    '''
    input: tile containing a single colony of interest and corresponding mask indicating segmentation
    returns: list containing variance of pixel intensities in colony, complexity score (sum of second derivative of colony divided by area), and fraction of colony pixels falling into each of 10 local binary patterns
    '''
    if len(np.shape(tile))==3:
        gray_tile = skimage.color.rgb2gray(tile)
    else:
        gray_tile = tile.copy()
    area = np.sum(1*mask)                 
    selem = np.array([[0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 1, 1, 1, 0, 0],
                      [0, 1, 1, 1, 1, 1, 0],
                      [1, 1, 1, 1, 1, 1, 1],
                      [0, 1, 1, 1, 1, 1, 0],
                      [0, 0, 1, 1, 1, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0]])
       
    if area > 0:
        # -- variance
        variance = np.var(tile[mask])
        # -- local binary patterns
        LBP = mask*skimage.feature.local_binary_pattern(tile, 8, 10, method='uniform') 
        hist, hist_centers = histogram(LBP[mask])
        temp = hist[hist>0]/area
        if len(temp) == 10:
                LBP_score = temp
        else:
            LBP_score = np.array(10*[np.nan])
        # sobel filter
        clean_tile = tile*mask
        mask_temp = clean_tile>0.2
        im = 255*clean_tile*mask_temp
        M = skimage.morphology.binary_erosion(clean_tile*mask_temp,footprint=selem )
        im_contrast = skimage.exposure.adjust_gamma(im)
        filt1 = skimage.filters.sobel(im, mask=M)
        filt2 = skimage.filters.sobel(filt1, mask=M)
        C_score = np.sum(filt2[mask])/area
    else:
        variance = np.nan
        LBP_score = np.array(10*[np.nan])
        C_score = np.nan
    return [variance, C_score]+LBP_score.tolist()

def BuildDF(tiles, masks, features ='size'):
    '''
    input: list of tiles to be included in data frame, list of masks segmenting the colony in those tiles, features = 'size' to include area, pixel sum, etc. or features = 'all' to include texture features as well.
    output: pandas dataframe with each row corresponding to a tile and each column corresponding to a feature. First column is "position" on the plate.
    '''
    
    positions = ([x+y for x in ['A','B','C','D','E','F','G','H'] 
                      for y in ['1','2','3','4','5','6','7','8','9','10','11','12']])    
    data = [None]*len(tiles)
    if features == 'all':
        features = (['Position','Area','AvgInt','PixelSum', 'Background', 'Perimeter','Variance', 'Complexity',
                    'LBP1', 'LBP2','LBP3','LBP4','LBP5','LBP6','LBP7','LBP8','LBP9','LBP10'])
        
        for j, (tile,mask,position) in enumerate(zip(tiles,masks,positions)):
            size_features = SizeFeatures(tile,mask)
            size_features.insert(0,position)
            texture_features = TextureFeatures(tile,mask)
            size_features.extend(texture_features)    
            data[j] = size_features.copy()
    elif features == 'size':
        features = ['Position','Area','AvgInt','PixelSum', 'Background', 'Perimeter']
        for j, (tile,mask,position) in enumerate(zip(tiles,masks,positions)):
            size_features = SizeFeatures(tile,mask)
            size_features.insert(0,position)    
            data[j] = size_features.copy()
    else:
        print('Feature options are \"size\" or \"all\"')
                  
    df = pd.DataFrame(data, columns = features)
    return df

def ProcessImage(file, sourcefolder, outputfolder,
                 crop_method = 'Auto', crop_param = None, s = 200, array_dimensions = (8,12), adjust = True, rotate = False, 
                 pin_size = 25, features = 'size', save = True, display = False, calibrate = False):
    '''
    input:  
            REQUIRED
            - file
            - sourcefolder
            - outputfolder
            OPTIONAL
            - crop_method
            - pin_size
            - crop_param
            - s
            - features
            - array_dimensions
            - adjust
            - rotate
            - save
            - display
            - calibrate
    output: - dataframe, tiles, masks, corners
    '''
    image = PP.LoadImage(file, sourcefolder, rotate = rotate)
    
    if crop_method == 'Auto':
        tiles, corners = PP.AutoCrop(file, sourcefolder, rotate = rotate, 
                                  s = s, array_dimensions = array_dimensions, 
                                  adjust=adjust, display = display)    
    elif crop_method == 'Grid':
        if calibrate:
            not_accept = True
            while not_accept:
                PP.GridDisplay(image)
                c1 = int(input("Input the x position of the center of patch A1 in pixels and press enter."))
                r1 = int(input("Input the y position of the center of patch A1 in pixels and press enter."))
                clear_output()
                PP.EnhancedGridDisplay(image, (r1,c1),s,array_dimensions)
                user_input = input("Would you like to proceed? y/n")
                if user_input == 'y':
                    not_accept = False
                if user_input == 'n':
                    continue                 
            clear_output()
            crop_param = (r1,c1)            
        if crop_param:
            A1_location = crop_param
        else:
            A1_location = (650,600)
        tiles, corners = PP.GridCrop(file, sourcefolder, rotate = rotate, 
                                  A1_location = A1_location, s = s, array_dimensions = array_dimensions, 
                                  adjust=adjust, display = display)
    elif crop_method == 'Click':
        tiles, corners = PP.ClickCrop(file, sourcefolder, 
                                   s=s, array_dimensions = array_dimensions, 
                                   rotate = rotate, adjust = adjust)
    else:
        print('crop_method options are Auto, Grid, or Crop')
        
    background = CalculateBackground(tiles)
    masks = [SegmentTile(tile, background, pin_size=pin_size, threshold_method='otsu') for tile in tiles]
    
    image_label_overlay = BuildSegmentedImage(image,masks,corners)
    df = BuildDF(tiles, masks, features = features)
    
    if save:
        filename = file.split('.')[0]+'.csv'
        (df.to_csv(os.path.join(outputfolder, filename), index=False))
        img_seg = 255*(image_label_overlay[200:-200,200:-200,:]) 
        A = Image.fromarray(img_seg.astype('uint8'), 'RGB')
        A.save(os.path.join(outputfolder,file.split('.')[0]+'_seg.jpg'))
    
    if display:
        fig, axs = plt.subplots(1,1,figsize=(12,16))
        axs.imshow(image_label_overlay[200:-200,200:-200,:]);
        axs.axis('off')
        plt.show();
    return df, tiles, masks, corners
    
def Process1Image(file, sourcefolder, outputfolder,
                 corners, s = 200, array_dimensions = (8,12), adjust = True, rotate = False, 
                 pin_size = 25, features = 'size', save = True, display = False):
    '''
        input: 
            REQUIRED
            - file
            - sourcefolder
            - outputfolder
            - corners
            OPTIONAL
            - pin_size
            - s
            - features
            - array_dimensions
            - adjust
            - rotate
            - save
            - display
            - calibrate
    output: none --saves dataframe and segmented image if save = True
    '''
    # -- Load Image
    image = PP.LoadImage(file, sourcefolder, rotate = rotate)
    # -- Set Corners
    if adjust:
        new_corners = PP.AutoAdjust(image, corners, s = s, array_dimensions = array_dimensions)
        corners = PP.AutoAdjust(image, new_corners, s = s, array_dimensions = array_dimensions)
    else:
        corners = corners
    # -- Crop tiles    
    tiles = PP.MakeTiles(image, corners, s = s)        
    # -- Calculate Background value    
    background = CalculateBackground(tiles)
    # -- Segment tiles
    masks = [SegmentTile(tile, background, pin_size=pin_size, threshold_method='otsu') for tile in tiles]
    # -- Merge Segmented Plate image
    image_label_overlay = BuildSegmentedImage(image,masks,corners)
    # -- Extract Features
    df = BuildDF(tiles, masks, features = features)
    # -- Save Results
    if save:
        filename = file.split('.')[0]+'.csv'
        (df.to_csv(os.path.join(outputfolder, filename), index=False))
        img_seg = 255*(image_label_overlay[s:-s,s:-s,:]) 
        A = Image.fromarray(img_seg.astype('uint8'), 'RGB')
        A.save(os.path.join(outputfolder,file.split('.')[0]+'_seg.jpg'))
    # -- Display Segmented Image
    if display:
        fig, axs = plt.subplots(1,1,figsize=(12,16))
        axs.imshow(image_label_overlay[s:-s,s:-s,:]);
        axs.axis('off')
        plt.show();
    return None    
        
def ProcessBatch(sourcefolder, outputfolder,   
                crop_method = 'Auto', s = 200, array_dimensions = (8,12), adjust = True, rotate = False, crop_param = None,
                calibrate = True, pin_size = 25, features = 'size', save = True, display = False):
    '''
    input:
            REQUIRED
            -sourcefolder
            -outputfolder
            OPTIONAL
            -crop_method
            -tile side length
            -array dimensions
            - auto adjust
            - rotate
            - crop_param
            - calibrate
            - pin size
            - features to compute 
            - save -- True or False
            - display -- True or False
    output: none
    '''
    
    # -- Set Up Folders
    wd = os.getcwd()
    if save:
        if not os.path.isdir(outputfolder):
            os.mkdir(outputfolder)
    # -- List Files
    file_list = PP.ListFiles(sourcefolder)
    image_list = [x  for x in file_list if "jpg" or "JPG" or "JPEG" or "jpeg" in x]
    
    # -- Calibrate using first image
    if calibrate:
        if crop_param:
            file = image_list[0]
            df, tiles, masks, corners = ProcessImage(file, sourcefolder, outputfolder,
                                             crop_method = crop_method, crop_param = crop_param, s = s, 
                                             array_dimensions = array_dimensions, pin_size = pin_size,
                                             features = features, adjust = adjust, rotate = rotate, 
                                             save = False, display = True, calibrate = False)
            PP.DisplayTiles(tiles)
            user_input = input("Would you like to proceed? y/n")
        else:
            file = image_list[0]
            df, tiles, masks, corners = ProcessImage(file, sourcefolder, outputfolder,
                                             crop_method = crop_method, crop_param = crop_param, s = s, 
                                             array_dimensions = array_dimensions, pin_size = pin_size,
                                             features = features, adjust = adjust, rotate = rotate, 
                                             save = False, display = True, calibrate = True)
            PP.DisplayTiles(tiles)
            user_input = input("Would you like to proceed? y/n")
    else:
        file = image_list[0]
        df, tiles, masks, corners = ProcessImage(file, sourcefolder, outputfolder,
                                         crop_method = crop_method, crop_param = crop_param, s = s, 
                                         array_dimensions = array_dimensions, pin_size = pin_size,
                                         features = features, adjust = adjust, rotate = rotate, 
                                         save = False, display = False, calibrate = calibrate)
        user_input = 'y'
        
    # -- Process all images
    if user_input == 'y':
        lines = (['BATCH PROCESS PARAMETERS',
                'Crop method: '+crop_method, 
                'Tile side length: ' +str(s)+ ' pixels', 
                'Pin radius: '+str(pin_size)+ ' pixels', 
                'Array dimensions: '+str(array_dimensions),
                'Autoadjust: '+str(adjust), 
                'Rotate images: '+str(adjust),
                'Features: '+features,
                'SEGMENTATION PARAMETERS',
                'Minimum intensity: 0.035 + background',
                'Minimum area: np.pi*(pin_size+10)**2',
                'Minimum circle certainty: 0.42',
                'Maximum circle certainty: 0.92', 
                'CROPPING PARAMETERS',
                'Min object size in auto adjust: 2000 pixels',
                'Plate Height in auto crop: ~1700 pixels',
                'Plate intensity in auto crop: 0.5',
                'Row buffer in auto crop: 250',
                'Column buffer in auto crop: 175',
                'COMPLETION'])
        
        with open(os.path.join(outputfolder,'BatchParameters.txt'), 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')
                
        count = 0
        missed_images = []
        for file in image_list:
            if display:
                try:
                    print(file)
                    Process1Image(file, sourcefolder, outputfolder,
                     corners, s = s, array_dimensions = array_dimensions, pin_size = pin_size,
                     features = features, adjust = adjust, rotate = rotate, save = True, display = True)
                    count = count+1
                except:
                    missed_images.append(file)
                    continue
            else:
                try:
                    Process1Image(file, sourcefolder, outputfolder,
                     corners, s = s, array_dimensions = array_dimensions, pin_size = pin_size,
                     features = features, adjust = adjust, rotate = rotate, save = True, display = False)
                    count = count+1
                    print(file)
                except:
                    missed_images.append(file)
                    continue
                
        file_object = open(os.path.join(outputfolder,'BatchParameters.txt'), 'a')
        file_object.write(str(count)+' out of '+ str(len(file_list))+ ' images processed.')
        file_object.close()
        with open(os.path.join(outputfolder,'MissedImages.txt'), 'w') as f:
            for image in missed_images:
                f.write(image)
                f.write('\n')
    return None
    
 
    
    
## -- Functions for loading example images
def Funnel():
    '''
    input: none
    output: returns sample image from Funnel cross data set
    '''
    stream = pkg_resources.resource_stream(__name__, 'Data/Im1.jpg')
    return skimage.io.imread(stream)
def PSAT1():
    '''
    input: none
    output: returns sample image from PSAT1 data set
    '''
    stream = pkg_resources.resource_stream(__name__, 'Data/Im2.jpg')
    return skimage.io.imread(stream)
def OTC():
    '''
    input: none
    output: returns sample image from OTC data set
    '''
    stream = pkg_resources.resource_stream(__name__, 'Data/Im3.jpg')
    return skimage.io.imread(stream)

## -- Function for processing example images

def ProcessImageTest(image, outputfolder,crop_method = 'Auto', crop_param = None, s = 200, pin_size = 25, array_dimensions = (8,12),
                 features = 'all', adjust = True, rotate = False, save = True, display = False, calibrate = False):
    '''
    Function to illustrate output of ProcessImage with the test images included in the package
    input: 
            REQUIRED
            -image (array)
            -outputfolder 
            OPTIONAL
            -crop_method
            -crop_param
            -s
            -pin_size
            -array_dimensions
            -features
            -adjust
            -rotate
            -save
            -display
            -calibrate            
    output: returns data frame, tiles, masks, corners
    '''
    if crop_method == 'Auto':
        tiles, corners = PP.AutoCropTest(image, s = s, array_dimensions = array_dimensions, 
                                  adjust=adjust, display = display)    
    elif crop_method == 'Grid':
        if calibrate:
            not_accept = True
            while not_accept:
                PP.GridDisplay(image)
                c1 = int(input("Input the x position of the center of patch A1 in pixels and press enter."))
                r1 = int(input("Input the y position of the center of patch A1 in pixels and press enter."))
                clear_output()
                PP.EnhancedGridDisplay(image, (r1,c1),s,array_dimensions)
                user_input = input("Would you like to proceed? y/n")
                if user_input == 'y':
                    not_accept = False
                if user_input == 'n':
                    continue                 
            clear_output()
            crop_param = (r1,c1)            
        if crop_param:
            A1_location = crop_param
        else:
            A1_location = (650,600)
        tiles, corners = PP.GridCropTest(image,
                                  A1_location = A1_location, s = s, array_dimensions = array_dimensions, 
                                  adjust=adjust, display = display)
    else:
        print('crop_method options are Auto or Grid')
        
    background = CalculateBackground(tiles)
    masks = [SegmentTile(tile, background, pin_size=pin_size, threshold_method='otsu') for tile in tiles]
    
    image_label_overlay = BuildSegmentedImage(image,masks,corners)
    df = BuildDF(tiles, masks, features =features,save=True)
    
    if save:
        filename = 'TestImage.csv'
        (df.to_csv(os.path.join(outputfolder, filename), index=False))
        img_seg = 255*(image_label_overlay[200:-200,200:-200,:]) 
        A = Image.fromarray(img_seg.astype('uint8'), 'RGB')
        A.save(os.path.join(outputfolder,'TestImage_seg.jpg'))
    
    if display:
        fig, axs = plt.subplots(1,1,figsize=(12,16))
        axs.imshow(image_label_overlay[200:-200,200:-200,:]);
        axs.axis('off')
        plt.show();
    return df, tiles, masks, corners

## -- Function for parallel processing batches of images 
   
def ParallelProcessBatch(sourcefolder, outputfolder, numThreads = 2,   
                crop_method = 'Auto', crop_param = None, s = 200, array_dimensions = (8,12), adjust = True, rotate = False,
                pin_size = 25, features = 'size', save = True, display = False, calibrate = True):
    '''
    Function to process a batch of images in parallel
            REQUIRED
            -sourcefolder
            -outputfolder 
            OPTIONAL
            -numThreads
            -crop_method
            -crop_param
            -s
            -pin_size
            -array_dimensions
            -features
            -adjust
            -rotate
            -save
            -display
            -calibrate            
    output: none
    '''
    
    # -- Define workhorse function with input arguments
    def workMethod(tno, workQueue):
        count = 0
        missed_images = []
        for item in workQueue:
            # ... process item ... #
            print("[Thread %d] Processing item '%s'" % (tno, str(item)))
            try:
                Process1Image(str(item), sourcefolder, outputfolder,
                             corners, s = s, pin_size = pin_size, array_dimensions = array_dimensions,
                             features = features, adjust = adjust, rotate = rotate, save = True, display = False)
                count = count + 1
            except:
                missed_images.append(file)
                continue
        print("[Thread %d] Processed '%s' out of '%s' images" % (tno, str(count),str(len(workQueue))))
        file_object = open(os.path.join(outputfolder,'BatchParameters.txt'), 'a')
        file_object.write(str(count)+' out of '+ str(len(workQueue))+ ' images processed on thread ' + str(tno)+'.')
        file_object.write('\n')
        file_object.close()
        
        with open(os.path.join(outputfolder,'MissedImages.txt'), 'w') as f:
            for image in missed_images:
                f.write(image)
                f.write('\n')
        return None

    
    # -- Set Up Folders
    wd = os.getcwd()
    if save:
        if not os.path.isdir(outputfolder):
            os.mkdir(outputfolder)
    # -- List Files
    file_list = PP.ListFiles(sourcefolder)
    image_list = [x  for x in file_list if "jpg" or "JPG" or "JPEG" or "jpeg" in x]
    
    # -- Calibrate using first image
    if calibrate:
        if crop_param:
            file = image_list[0]
            df, tiles, masks, corners = ProcessImage(file, sourcefolder, outputfolder,
                                             crop_method = crop_method, crop_param = crop_param, s = s, 
                                             array_dimensions = array_dimensions, pin_size = pin_size,
                                             features = features, adjust = adjust, rotate = rotate, 
                                             save = False, display = True, calibrate = False)
            PP.DisplayTiles(tiles)
            user_input = input("Would you like to proceed? y/n")
        else:
            file = image_list[0]
            df, tiles, masks, corners = ProcessImage(file, sourcefolder, outputfolder,
                                             crop_method = crop_method, crop_param = crop_param, s = s, 
                                             array_dimensions = array_dimensions, pin_size = pin_size,
                                             features = features, adjust = adjust, rotate = rotate, 
                                             save = False, display = True, calibrate = True)
            PP.DisplayTiles(tiles)
            user_input = input("Would you like to proceed? y/n")
    else:
        file = image_list[0]
        df, tiles, masks, corners = ProcessImage(file, sourcefolder, outputfolder,
                                         crop_method = crop_method, crop_param = crop_param, s = s, 
                                         array_dimensions = array_dimensions, pin_size = pin_size,
                                         features = features, adjust = adjust, rotate = rotate, 
                                         save = False, display = False, calibrate = calibrate)
        user_input = 'y'
        
    # -- Process all images
    if user_input == 'y':
        lines = (['BATCH PROCESS PARAMETERS',
                'Crop method: '+crop_method, 
                'Tile side length: ' +str(s)+ ' pixels', 
                'Pin radius: '+str(pin_size)+ ' pixels', 
                'Array dimensions: '+str(array_dimensions),
                'Autoadjust: '+str(adjust), 
                'Rotate images: '+str(adjust),
                'Features: ' +features,
                'SEGMENTATION PARAMETERS',
                'Minimum intensity: 0.035 + background',
                'Minimum area: np.pi*(pin_size+10)**2',
                'Minimum circle certainty: 0.42',
                'Maximum circle certainty: 0.92', 
                'CROPPING PARAMETERS',
                'Min object size in auto adjust: 2000 pixels',
                'Plate Height in auto crop: ~1700 pixels',
                'Plate intensity in auto crop: 0.5',
                'Row buffer in auto crop: 250',
                'Column buffer in auto crop: 175',
                'COMPLETION'])
        
        with open(os.path.join(outputfolder,'BatchParameters.txt'), 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')
                
        ## Break image_list up into chunks for each thread and assign workMethod
        threads = []
        for tno in range(0, numThreads-1):
            # divide the allWork among the threads, and make sure not to miss anything ...
            chunkSize = int(len(image_list)/numThreads)
            workStart = chunkSize*tno
            workEnd = chunkSize*(tno+1)
            threads.append(threading.Thread(target=workMethod, args=(tno, image_list[workStart:workEnd])))
            threads[-1].start()
        # ... last thread takes the leftovers    
        workStart = chunkSize*(numThreads-1)
        workEnd = len(image_list)
        threads.append(threading.Thread(target=workMethod, args=(numThreads-1, image_list[workStart:workEnd])))
        threads[-1].start()
            

        # wait for each thread to finish its work, the script will not progress until all work is done
        for thd in threads:
            thd.join()

    return None
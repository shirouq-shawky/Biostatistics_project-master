import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pyzbar.pyzbar import decode
from PIL.ExifTags import TAGS

# -----------------------------
# Functions
# -----------------------------

def get_field (exif,field):
    for (k,v) in exif.items():
        if TAGS.get(k) == field:
            return v

def Rename(image_folder_path, mastersheet, columns, code = '', delimiter = '_', datetime_delimiter = '-', datetime = True):
    '''
    Input: -image_folder_path: file path to folder containing images to be renamed inplace. eg. r'C:User\\Documents\\Image_Folder'
           -mastersheet: file name of .xlsx or .csv containing the information to be used in the new file name. The barcode column must come             first and it must have the title 'Barcode'
           -columns: list strings indicating the column titles of which columns contain information to be included in the name, eg.
           ['Plate', 'Condition', 'Replicate']
           -code: Additional placeholder for other info to be included in the name. Could be short hand for the folder of images eg '24h'. 
           Default is blank.
    Output: None
            -datetime indicates whether the timestamp should be included in the image name. The default is True, but it can be switched to
            False
    
    Running this function with the master sheet in the current working directory will result in renaming the 
    .jpg files within the image_folder_path as 'col1_col2_...colN_code_hr-min-sec.jpg'. The appropriate strings from each column are
    determined by decoding the barcode within the image and matching the result with the appropriate row in mastersheet. Instead of underscores, a different delimiter can be set with the kwarg delimiter, and instead of hyphens a different delimiter for the date-time info can be set with kwarg datetime_delimiter
    '''
    if 'csv' in mastersheet:
        df = pd.read_csv(mastersheet)
    else:
        df = pd.read_excel (mastersheet)
    os.chdir(image_folder_path)
    s = df['Barcode']
    image_filenames = glob.glob('*.jpg')
    
    for filename in image_filenames:
        try:
            image = Image.open(filename)
            im = np.array(image)
            all_the_info = decode(im)
            if not all_the_info:
                print('Barcode could not be detected and/or decoded for image ' + filename)
                continue
            barcode = all_the_info[0].data
            d = barcode.decode('ASCII')
            image_row = df.loc[s==int(d)]
            info = image_row.iloc[0]
            temp = []
            for c in columns:
                temp.append(info[c])
            if len(code)>0:
                temp.append(code)
            if datetime:
                exif_data = image._getexif()
                DT = get_field(exif_data,'DateTime')
                Time = DT.split()[1]
                Time = Time.replace(':',datetime_delimiter)
                temp.append(Time)

            new_name = delimiter.join(temp) + '.jpg'
            print(new_name)
            dst = new_name
            src = filename
            os.rename(src,dst)
        except:
            print('Error processing image ' + filename + '. Possibly barcode ' + d+ 'not found in mastersheet.')
            continue
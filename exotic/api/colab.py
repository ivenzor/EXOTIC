# ########################################################################### #
#    Copyright (c) 2019-2020, California Institute of Technology.
#    All rights reserved.  Based on Government Sponsored Research under
#    contracts NNN12AA01C, NAS7-1407 and/or NAS7-03001.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions
#    are met:
#      1. Redistributions of source code must retain the above copyright
#         notice, this list of conditions and the following disclaimer.
#      2. Redistributions in binary form must reproduce the above copyright
#         notice, this list of conditions and the following disclaimer in
#         the documentation and/or other materials provided with the
#         distribution.
#      3. Neither the name of the California Institute of
#         Technology (Caltech), its operating division the Jet Propulsion
#         Laboratory (JPL), the National Aeronautics and Space
#         Administration (NASA), nor the names of its contributors may be
#         used to endorse or promote products derived from this software
#         without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE CALIFORNIA
#    INSTITUTE OF TECHNOLOGY BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
#    TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# ########################################################################### #
#    EXOplanet Transit Interpretation Code (EXOTIC)
#    # NOTE: See companion file version.py for version info.
# ########################################################################### #
# IMPORTANT RUNTIME INFORMATION ABOUT DATA ACCESS AND STORAGE
#
# If the user presses enter to run the sample data, download sample data if
# needed and put it into a sample-data directory at the top level of the
# user's Gdrive. Count the .fits files (images) and .json files (inits files)
# in the directory entered by the user (or in the sample-data directory if the
# user pressed enter).  If there are at least 20 .fits files, assume this is a
# directory of images and display the first one in the series. If there is
# exactly one inits file in the directory, show the specified target and comp
# coords so that the user can check these against the displayed image.
# Otherwise, prompt for target / comp coords and make an inits file based on
# those (save this new inits file in the folder with the output files so that
# the student can consult it later).  Finally, run EXOTIC with the newly-made
# or pre-existing inits file, plus any other inits files in the directory.
#
#########################################################
from astropy.io import fits
from astropy.time import Time
from barycorrpy import utc_tdb
# import bokeh.io
# from bokeh.io import output_notebook
from bokeh.palettes import Viridis256
from bokeh.plotting import figure, output_file, show
from bokeh.models import BoxZoomTool, ColorBar, FreehandDrawTool, HoverTool, LinearColorMapper, LogColorMapper, \
  LogTicker, PanTool, ResetTool, WheelZoomTool
# import copy
from io import BytesIO
from IPython.display import display, HTML
# from IPython.display import Image
# from ipywidgets import widgets, HBox
import json
import numpy as np
import os
from pprint import pprint
import re
from scipy.ndimage import label
from skimage.transform import rescale, resize, downscale_local_mean
# import subprocess
import time


from bokeh.plotting import figure, show
import numpy as np
from astropy.io import fits
from skimage.transform import resize
from bokeh.models import (ColumnDataSource, LogColorMapper, HoverTool, CustomJS, 
                         Button, Div, FreehandDrawTool, Slider, PanTool, 
                         BoxZoomTool, WheelZoomTool, ResetTool)
from bokeh.layouts import column, row

def display_image(filename):
    """
    Downsampling if needed with options to flip and rotate while maintaining original coordinates.
    """
    MAX_MEGAPIXELS = 1.0
    MAX_DISPLAY_DIMENSION = 500
    
    with fits.open(filename) as hdu:
        extension = 0
        while hdu[extension].header["NAXIS"] == 0:
            extension += 1
        
        original_data = hdu[extension].data
        original_shape = original_data.shape
        megapixels = (original_shape[0] * original_shape[1]) / 1_000_000

        if megapixels > MAX_MEGAPIXELS:
            scale_factor = np.sqrt(MAX_MEGAPIXELS / megapixels)
            new_shape = tuple(int(dim * scale_factor) for dim in original_shape)
            print(f"Downsampling image from {megapixels:.2f} MP to {MAX_MEGAPIXELS} MP")
            data = resize(original_data, new_shape, preserve_range=True).astype(original_data.dtype)
            scale_y = original_shape[0] / new_shape[0]
            scale_x = original_shape[1] / new_shape[1]
        else:
            data = original_data.copy()
            scale_y = 1
            scale_x = 1

    # Create all rotated versions
    data_rot90 = np.rot90(data)
    data_rot180 = np.rot90(data, 2)
    data_rot270 = np.rot90(data, 3)

    h, w = data.shape
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    orig_x = (x_coords * scale_x).astype(int)
    orig_y = (y_coords * scale_y).astype(int)

    # Store coordinates as 1D arrays
    orig_x_flat = orig_x.ravel()
    orig_y_flat = orig_y.ravel()
    values_flat = data.ravel()

    # Create coordinate mappings for each transformation
    coord_versions = {
        0: {
            'normal': (orig_x, orig_y),
            'h': (np.fliplr(orig_x), orig_y),
            'v': (orig_x, np.flipud(orig_y)),
            'hv': (np.fliplr(orig_x), np.flipud(orig_y))
        },
        90: {
            'normal': (np.rot90(orig_x), np.rot90(orig_y)),
            'h': (np.fliplr(np.rot90(orig_x)), np.rot90(orig_y)),
            'v': (np.rot90(orig_x), np.flipud(np.rot90(orig_y))),
            'hv': (np.fliplr(np.rot90(orig_x)), np.flipud(np.rot90(orig_y)))
        },
        180: {
            'normal': (np.rot90(orig_x, 2), np.rot90(orig_y, 2)),
            'h': (np.fliplr(np.rot90(orig_x, 2)), np.rot90(orig_y, 2)),
            'v': (np.rot90(orig_x, 2), np.flipud(np.rot90(orig_y, 2))),
            'hv': (np.fliplr(np.rot90(orig_x, 2)), np.flipud(np.rot90(orig_y, 2)))
        },
        270: {
            'normal': (np.rot90(orig_x, 3), np.rot90(orig_y, 3)),
            'h': (np.fliplr(np.rot90(orig_x, 3)), np.rot90(orig_y, 3)),
            'v': (np.rot90(orig_x, 3), np.flipud(np.rot90(orig_y, 3))),
            'hv': (np.fliplr(np.rot90(orig_x, 3)), np.flipud(np.rot90(orig_y, 3)))
        }
    }

    # Pre-compute all flipped versions for each rotation
    flipped_versions = {
        0: {
            'normal': data,
            'h': np.fliplr(data),
            'v': np.flipud(data),
            'hv': np.flipud(np.fliplr(data))
        },
        90: {
            'normal': data_rot90,
            'h': np.fliplr(data_rot90),
            'v': np.flipud(data_rot90),
            'hv': np.flipud(np.fliplr(data_rot90))
        },
        180: {
            'normal': data_rot180,
            'h': np.fliplr(data_rot180),
            'v': np.flipud(data_rot180),
            'hv': np.flipud(np.fliplr(data_rot180))
        },
        270: {
            'normal': data_rot270,
            'h': np.fliplr(data_rot270),
            'v': np.flipud(data_rot270),
            'hv': np.flipud(np.fliplr(data_rot270))
        }
    }

    # Pre-calculate percentile values
    lower_percentiles = np.arange(0, 95, 1)
    upper_percentiles = np.arange(95, 100.1, 0.1)
    percentiles = np.concatenate([lower_percentiles, upper_percentiles])
    percentile_values = np.percentile(data, percentiles)
    slider_to_idx = np.linspace(0, len(percentile_values)-1, 101).astype(int)
    
    initial_low_percentile = 55
    initial_high_percentile = 99
    initial_low_idx = slider_to_idx[initial_low_percentile]
    initial_high_idx = slider_to_idx[initial_high_percentile]
    initial_low = percentile_values[initial_low_idx]
    initial_high = percentile_values[initial_high_idx]

    color_mapper = LogColorMapper(palette="Cividis256", 
                                low=initial_low,
                                high=initial_high)

    source = ColumnDataSource(data={
        'image': [data],
        'x': [0],
        'y': [0],
        'dw': [original_shape[1]],
        'dh': [original_shape[0]],
        'flip_h': [False],
        'flip_v': [False],
        'rotation': [0],
        'scale_x': [scale_x],
        'scale_y': [scale_y],
        'percentile_values': [percentile_values.tolist()],
        'slider_to_idx': [slider_to_idx.tolist()],
        'orig_x_coords': [orig_x],
        'orig_y_coords': [orig_y]
    })

    # Add all versions to source
    for angle in [0, 90, 180, 270]:
        for flip_type in ['normal', 'h', 'v', 'hv']:
            key = f'rot{angle}_{flip_type}'
            source.data[key] = [flipped_versions[angle][flip_type]]
            source.data[f'{key}_x'] = [coord_versions[angle][flip_type][0]]
            source.data[f'{key}_y'] = [coord_versions[angle][flip_type][1]]

    get_orig_coords = CustomJS(code="""
        const cb_data = cb_obj.geometryData;
        if (!cb_data) return null;
        
        let x = Math.floor(cb_data.x);
        let y = Math.floor(cb_data.y);
        const w = source.data.width[0];
        const h = source.data.height[0];
        const rot = source.data.rotation[0];
        const flip_h = source.data.flip_h[0];
        const flip_v = source.data.flip_v[0];
        const scale_x = source.data.scale_x[0];
        const scale_y = source.data.scale_y[0];
        
        // Transform coordinates based on current state
        // First handle flips
        if (flip_h) {
            x = w - x - 1;
        }
        if (flip_v) {
            y = h - y - 1;
        }
        
        // Then handle rotation
        let tx = x;
        let ty = y;
        
        if (rot === 90) {
            x = h - ty - 1;
            y = tx;
        } else if (rot === 180) {
            x = w - tx - 1;
            y = h - ty - 1;
        } else if (rot === 270) {
            x = ty;
            y = w - tx - 1;
        }
        
        // Calculate index in flattened array
        const idx = y * w + x;
        
        // Get original coordinates
        const origX = x * scale_x;
        const origY = y * scale_y;
        const value = source.data.values[0][idx];
        
        return {
            x: Math.round(origX),
            y: Math.round(origY),
            value: value
        };
    """)

    hover = HoverTool(
        tooltips=[
            ("Original x", "@orig_x_coords{0}"),
            ("Original y", "@orig_y_coords{0}"),
            ("Value", "@image")
        ],
        mode='mouse',
        point_policy='snap_to_data'
    )

    aspect_ratio = data.shape[1] / data.shape[0]
    if aspect_ratio > 1:
        p_width = MAX_DISPLAY_DIMENSION
        p_height = int(p_width / aspect_ratio)
    else:
        p_height = MAX_DISPLAY_DIMENSION
        p_width = int(p_height * aspect_ratio)

    fig = figure(width=p_width, height=p_height,
                tools=[PanTool(), BoxZoomTool(), WheelZoomTool(), ResetTool(), hover],
                x_range=(0, original_shape[1]),
                y_range=(0, original_shape[0]))

    if megapixels > MAX_MEGAPIXELS:
        fig.title.text = f"Downsampled Image (Original: {megapixels:.2f} MP, Current: {MAX_MEGAPIXELS} MP)"
    else:
        fig.title.text = f"Original Image ({megapixels:.2f} MP)"
    
    fig.image(image='image', x='x', y='y', dw='dw', dh='dh',
              source=source, color_mapper=color_mapper)

    r = fig.multi_line('x', 'y', source={'x':[], 'y':[]}, color='white', line_width=3)
    fig.add_tools(FreehandDrawTool(renderers=[r]))

    # Create control buttons
    button_flip_h = Button(label="Flip Horizontal", button_type="default", width=120)  # Changed from primary to default
    button_flip_v = Button(label="Flip Vertical", button_type="default", width=120)    # Changed from primary to default
    button_rotate_left = Button(label="Rotate Left", button_type="default", width=120) # Changed from primary to default
    button_rotate_right = Button(label="Rotate Right", button_type="default", width=120) # Changed from primary to default
    button_reset = Button(label="Reset Flips and Rotations", button_type="warning", width=200)  # Changed text and width
    
    status_h = Div(text="Horizontal Flip: Off", width=120, styles={'color': 'gray', 'font-size': '12px'})
    status_v = Div(text="Vertical Flip: Off", width=120, styles={'color': 'gray', 'font-size': '12px'})
    status_rot = Div(text="Rotation: 0°", width=120, styles={'color': 'gray', 'font-size': '12px'})

    low_percentile = Slider(start=0, end=90, value=initial_low_percentile, step=1,
                          title="Low Percentile", width=200)
    high_percentile = Slider(start=95, end=100, value=initial_high_percentile, step=0.1,
                          title="High Percentile", width=200)

    update_image = CustomJS(args=dict(source=source, status_rot=status_rot), code="""
        const rot = source.data.rotation[0];
        const flip_h = source.data.flip_h[0];
        const flip_v = source.data.flip_v[0];
        
        let flip_type = 'normal';
        if (flip_h && flip_v) flip_type = 'hv';
        else if (flip_h) flip_type = 'h';
        else if (flip_v) flip_type = 'v';
        
        const key = 'rot' + rot + '_' + flip_type;
        source.data.image[0] = source.data[key][0];
        source.data.orig_x_coords[0] = source.data[key + '_x'][0];
        source.data.orig_y_coords[0] = source.data[key + '_y'][0];
        
        // Update rotation status
        status_rot.text = 'Rotation: ' + rot + '°';
        
        source.change.emit();
    """)

    rotate_left = CustomJS(args=dict(source=source, update_image=update_image), code="""
        source.data.rotation[0] = (source.data.rotation[0] + 90) % 360;
        update_image.execute();
    """)

    rotate_right = CustomJS(args=dict(source=source, update_image=update_image), code="""
        source.data.rotation[0] = (source.data.rotation[0] - 90 + 360) % 360;
        update_image.execute();
    """)

    flip_h_callback = CustomJS(args=dict(source=source, status=status_h, update_image=update_image, button=button_flip_h), code="""
        source.data.flip_h[0] = !source.data.flip_h[0];
        status.text = 'Horizontal Flip: ' + (source.data.flip_h[0] ? 'On' : 'Off');
        status.styles = {'color': source.data.flip_h[0] ? 'green' : 'gray', 'font-size': '12px'};
        button.button_type = source.data.flip_h[0] ? 'primary' : 'default';  // Change button color
        update_image.execute();
    """)

    flip_v_callback = CustomJS(args=dict(source=source, status=status_v, update_image=update_image, button=button_flip_v), code="""
        source.data.flip_v[0] = !source.data.flip_v[0];
        status.text = 'Vertical Flip: ' + (source.data.flip_v[0] ? 'On' : 'Off');
        status.styles = {'color': source.data.flip_v[0] ? 'green' : 'gray', 'font-size': '12px'};
        button.button_type = source.data.flip_v[0] ? 'primary' : 'default';  // Change button color
        update_image.execute();
    """)

    rotate_left = CustomJS(args=dict(source=source, update_image=update_image, button=button_rotate_left, button_right=button_rotate_right), code="""
        source.data.rotation[0] = (source.data.rotation[0] + 90) % 360;
        // Update button colors based on rotation state
        if (source.data.rotation[0] === 0) {
            button.button_type = 'default';
            button_right.button_type = 'default';
        } else {
            button.button_type = 'primary';
            button_right.button_type = 'primary';
        }
        update_image.execute();
    """)

    rotate_right = CustomJS(args=dict(source=source, update_image=update_image, button=button_rotate_right, button_left=button_rotate_left), code="""
        source.data.rotation[0] = (source.data.rotation[0] - 90 + 360) % 360;
        // Update button colors based on rotation state
        if (source.data.rotation[0] === 0) {
            button.button_type = 'default';
            button_left.button_type = 'default';
        } else {
            button.button_type = 'primary';
            button_left.button_type = 'primary';
        }
        update_image.execute();
    """)

    reset_callback = CustomJS(args=dict(source=source, 
                                    status_h=status_h, 
                                    status_v=status_v,
                                    status_rot=status_rot,
                                    low_percentile=low_percentile,
                                    high_percentile=high_percentile,
                                    color_mapper=color_mapper,
                                    update_image=update_image,
                                    button_h=button_flip_h,
                                    button_v=button_flip_v,
                                    button_left=button_rotate_left,
                                    button_right=button_rotate_right), 
                            code="""
        source.data.flip_h[0] = false;
        source.data.flip_v[0] = false;
        source.data.rotation[0] = 0;
        
        // Reset all button colors
        button_h.button_type = 'default';
        button_v.button_type = 'default';
        button_left.button_type = 'default';
        button_right.button_type = 'default';
        
        status_h.text = 'Horizontal Flip: Off';
        status_v.text = 'Vertical Flip: Off';
        status_rot.text = 'Rotation: 0°';
        status_h.styles = {'color': 'gray', 'font-size': '12px'};
        status_v.styles = {'color': 'gray', 'font-size': '12px'};
        
        low_percentile.value = 55;
        high_percentile.value = 99;
        
        const percentile_values = source.data.percentile_values[0];
        const slider_to_idx = source.data.slider_to_idx[0];
        const low_idx = slider_to_idx[55];
        const high_idx = slider_to_idx[99];
        
        color_mapper.low = percentile_values[low_idx];
        color_mapper.high = percentile_values[high_idx];
        
        update_image.execute();
    """)

    update_percentiles = CustomJS(args=dict(source=source,
                                          color_mapper=color_mapper,
                                          low_percentile=low_percentile,
                                          high_percentile=high_percentile), 
                                code="""
        const percentile_values = source.data.percentile_values[0];
        const slider_to_idx = source.data.slider_to_idx[0];
        const low_idx = slider_to_idx[Math.round(low_percentile.value)];
        const high_idx = slider_to_idx[Math.round(high_percentile.value * 10) / 10];
        
        color_mapper.low = percentile_values[low_idx];
        color_mapper.high = percentile_values[high_idx];
    """)

    # Attach callbacks
    button_flip_h.js_on_click(flip_h_callback)
    button_flip_v.js_on_click(flip_v_callback)
    button_rotate_left.js_on_click(rotate_left)
    button_rotate_right.js_on_click(rotate_right)
    button_reset.js_on_click(reset_callback)
    low_percentile.js_on_change('value', update_percentiles)
    high_percentile.js_on_change('value', update_percentiles)

    # Create layout
    percentile_controls = row(low_percentile, high_percentile, spacing=20)
    button_controls = row(
        column(button_flip_h, status_h),
        column(button_flip_v, status_v),
        column(button_rotate_left),
        column(button_rotate_right, status_rot),
        column(button_reset),
        spacing=10
    )
    layout = column(percentile_controls, button_controls, fig)

    show(layout)

#########################################################

def floats_to_ints(l):
  while (True):
#    print (l)
    m = re.search(r"^(.*?)(\d+\.\d+)(.*?)$", l)
    if m:
      start, fl, end = m.group(1), float(m.group(2)), m.group(3)
      l = start+str("%.0f" % fl)+end
    else:
      return(l)
  
#########################################################

# Find a field in the image fits header or prompt the user to enter the corresponding
# value.

def check_dir(p):
  p = p.replace("\\", "/")

  if not(os.path.isdir(p)):
    print(f"Problem: the directory {p} doesn't seem to exist")
    print("on your Gdrive filesystem.")
    return("")
  return(p)

#########################################################

def add_sign(var):
  str_var = str(var)
  m=re.search(r"^[\+\-]", str_var)
  if m:
    return(str_var)
  if float(var) >= 0:
    return(str("+%.6f" % float(var)))
  else:
    return(str("-%.6f" % float(var)))

#########################################################

def get_val(hdr, ks):
  for key in ks:
    if key in hdr.keys():
      return hdr[key]
    if key.lower() in hdr.keys():
      return hdr[key.lower()]
    new_key = key[0]+key[1:len(key)].lower()  # first letter capitalized
    if new_key in hdr.keys():
      return hdr[new_key]
  return("")

#########################################################

def process_lat_long(val, key):
  m = re.search(r"\'?([+-]?\d+)[\s\:](\d+)[\s\:](\d+\.?\d*)", val)
  if m:
    deg, min, sec = float(m.group(1)), float(m.group(2)), float(m.group(3))
    if deg < 0:
      v = deg - (((60*min) + sec)/3600)
    else:
      v = deg + (((60*min) + sec)/3600)
    return(add_sign(v))
  m = re.search("^\'?([+-]?\d+\.\d+)", val)
  if m:
    v = float(m.group(1))
    return(add_sign(v))
  else:
    print(f"Cannot match value {val}, which is meant to be {key}.")

#########################################################

# Convert a MicroObservatory timestamp (which is in local time) to UTC.

def convert_Mobs_to_utc(datestamp, latitude, longitude, height):

#  print(datestamp)
  t = Time(datestamp[0:21], format='isot', scale='utc')
  t -= 0.33

  return(str(t)[0:10])

#########################################################

def find (hdr, ks, obs):
  # Special stuff for MObs and Boyce-Astro Observatories
  boyce = {"FILTER": "ip", "LATITUDE": "+32.6135", "LONGITUD": "-116.3334", "HEIGHT": 1405 }
  mobs = {"FILTER": "V", "LATITUDE": "+37.04", "LONGITUD": "-110.73", "HEIGHT": 2606 }

  if "OBSERVAT" in hdr.keys() and hdr["OBSERVAT"] == 'Whipple Observatory':
    obs = "MObs"

#  if "USERID" in hdr.keys() and hdr["USERID"] == 'PatBoyce':
#    obs = "Boyce"

  if obs == "Boyce":
    boyce_val = get_val(boyce, ks)
    if (boyce_val != ""):
      return(boyce_val)
  if obs == "MObs":
    mobs_val = get_val(mobs, ks)
    if (mobs_val != ""):
      return(mobs_val)

  val = get_val(hdr, ks)

  if ks[0] == "LATITUDE" and val != "":         # Because EXOTIC needs these with signs shown.
    return(process_lat_long(str(val), "latitude"))
  if ks[0] == "LONGITUD" and val != "":
    return(process_lat_long(str(val), "longitude"))

  if (val != ""):
    return(val)

  print(f"\nI cannot find a field with any of these names in your image header: \n{ks}.")
  print("Please enter the value (not the name of the header field, the actual value) that should")
  print("be used for the value associated with this field.\n")
  if ks[0] == "HEIGHT":
    print("The units of elevation are meters.")
  
  value = input("")

  return(value)

###############################################

def look_for_calibration(image_dir):
  darks_dir, flats_dir, biases_dir = "null", "null", "null"

  m = re.search(r"(.*?)(\d\d\d\d\-\d\d\-\d\d)\/images", image_dir)  # This handles the way I set up the MObs image paths for my seminar teams.
  if m:
    prefix, date = m.group(1), m.group(2)
    darks = prefix+date+"/darks"
    if os.path.isdir(darks):
      darks_dir = str("\""+darks+"\"")
      
  d_names = ["dark", "darks", "DARK", "DARKS", "Dark", "Darks"]  # Possible names for calibration image directories.
  f_names = ["flat", "flats", "FLAT", "FLATS", "Flat", "Flats"]
  b_names = ["bias", "biases", "BIAS", "BIASES", "Bias", "Biases"]

  for d in d_names:
    if os.path.isdir(os.path.join(image_dir, d)):
      darks_dir = str("\""+os.path.join(image_dir, d)+"\"")
      break

  for f in f_names:
    if os.path.isdir(os.path.join(image_dir, f)):
      flats_dir = str("\""+os.path.join(image_dir, f)+"\"")
      break

  for b in b_names:
    if os.path.isdir(os.path.join(image_dir, b)):
      biases_dir = str("\""+os.path.join(image_dir, b)+"\"")
      break

  return(darks_dir, flats_dir, biases_dir)

###############################################

# Writes a new inits file into the directory with the output plots.  This prompts
# for needed information that it cannot find in the fits header of the first image.

def make_inits_file(planetary_params, image_dir, output_dir, first_image, targ_coords, comp_coords, obs, aavso_obs_code, sec_obs_code, sample_data):
  inits_file_path = output_dir+"inits.json"
  hdul = fits.open(first_image)

  extension = 0
  hdr = fits.getheader(filename=first_image, ext=extension)
  while hdr['NAXIS'] == 0:
    extension += 1
    hdr = fits.getheader(filename=first_image, ext=extension)

  min, max = "null", "null"
  filter = find(hdr, ['FILTER', 'FILT'], obs)
  if filter == "w":
    filter = "PanSTARRS-w"
    min = "404"
    max = "846"
  if filter == "Clear":
    filter = "V"
  if filter == "ip":
    min = "690"
    max = "819"
  if filter == "EXO":
    filter = "CBB"
  if re.search(r"Green", filter, re.IGNORECASE):
    filter = "SG"
    
  date_obs = find(hdr,["DATE", "DATE_OBS", "DATE-OBS"], obs)
  date_obs = date_obs.replace("/", "_")
  longitude = find(hdr,['LONGITUD', 'LONG', 'LONGITUDE', 'SITELONG'],obs)
  latitude = find(hdr,['LATITUDE', 'LAT', 'SITELAT'],obs)
  height = float(find(hdr, ['HEIGHT', 'ELEVATION', 'ELE', 'EL', 'OBSGEO-H', 'ALT-OBS', 'SITEELEV'], obs))
  obs_notes = "N/A"

  mobs_data = False
  # For MObs, the date is local rather than UTC, so convert.
  if "OBSERVAT" in hdr.keys() and hdr["OBSERVAT"] == 'Whipple Observatory':
    date_obs = convert_Mobs_to_utc(date_obs, latitude, longitude, height)
    weather = hdr["WEATHER"] 
    temps = float(hdr["TELTEMP"]) - float(hdr["CAMTEMP"])
    obs_notes = str("First image seeing %s (0: poor, 100: excellent), Teltemp - Camtemp %.1f.  These observations were conducted with MicroObservatory, a robotic telescope network managed by the Harvard-Smithsonian Center for Astrophysics on behalf of NASA's Universe of Learning. This work is supported by NASA under award number NNX16AC65A to the Space Telescope Science Institute." % (weather, temps))
    sec_obs_code = "MOBS"  
    mobs_data = True
  
  if aavso_obs_code == "":
      aavso_obs_code = "N/A"
  if sec_obs_code == "":
      sec_obs_code = "N/A"

  obs_date = date_obs[0:10]
  (darks_dir, flats_dir, biases_dir) = look_for_calibration(image_dir)

  with open(inits_file_path, 'w') as inits_file:
    inits_file.write("""
{
  %s,
    "user_info": {
            "Directory with FITS files": "%s",
            "Directory to Save Plots": "%s",
            "Directory of Flats": %s,
            "Directory of Darks": %s,
            "Directory of Biases": %s,

            "AAVSO Observer Code (N/A if none)": "%s",
            "Secondary Observer Codes (N/A if none)": "%s",

            "Observation date": "%s",
            "Obs. Latitude": "%s",
            "Obs. Longitude": "%s",
            "Obs. Elevation (meters)": %d,
            "Camera Type (CCD or DSLR)": "CCD",
            "Pixel Binning": "1x1",
            "Filter Name (aavso.org/filters)": "%s",
            "Observing Notes": "%s",

            "Plate Solution? (y/n)": "y",
            "Add Comparison Stars from AAVSO? (y/n)": "n",

            "Target Star X & Y Pixel": %s,
            "Comparison Star(s) X & Y Pixel": %s,
            
            "Demosaic Format": null,
            "Demosaic Output": null
    },    
    "optional_info": {
            "Pixel Scale (Ex: 5.21 arcsecs/pixel)": null,
            "Filter Minimum Wavelength (nm)": %s,
            "Filter Maximum Wavelength (nm)": %s
    }
}
""" % (planetary_params, image_dir, output_dir, flats_dir, darks_dir, biases_dir, 
       aavso_obs_code, sec_obs_code, obs_date, latitude, longitude, height, filter, 
       obs_notes, targ_coords, comp_coords, min, max))

  display(HTML('<p class="output"><b>Initialization File Created.</b></p>'))
  print(f'Created: {inits_file_path}')
  print('This folder will also contain the output files when EXOTIC finishes running.')

  if not mobs_data:  
    print(f"\nThe inits.json file currently says that your observatory latitude was {latitude} deg,")
    print(f"longitude was {longitude} deg, and elevation was {height}m.  \n")
    print("*** If any of these are incorrect, please change them in the inits.json file. ***")
    print("*** (Please make sure that Western longitudes have a negative sign! ***")
    print("*** TheSkyX sometimes stamps Western longitudes as positive; this needs to be switched! ***\n")

  display(HTML('<p class="output"><br /><b>If you want to change anything in the inits file, such as planetary parameters or user info, please do that now.</b></p>'))
  print('You can edit the file by clicking the folder icon in the left nav,')
  print(f'navigating to the inits file at {inits_file_path}, and double-clicking the file.')
  print('\nWhen you are done, save your changes, and proceed to the next step.')
  

  return(inits_file_path)
  
##############################################################

def fix_planetary_params (p_param_dict):
  for param in p_param_dict.keys():
    if param == "Target Star RA" or param == "Target Star Dec" or param == "Planet Name" or param == "Host Star Name" or param == "Argument of Periastron (deg)":
      continue
    val = p_param_dict[param]
    if val == 0.0 or np.isnan(float(val)):
      if param == "Orbital Eccentricity (0 if null)":
        continue
      if param == "Ratio of Planet to Stellar Radius (Rp/Rs)":
        p_param_dict[param] = 0.151
      if param == "Ratio of Planet to Stellar Radius (Rp/Rs) Uncertainty":
        p_param_dict[param] = 0.151
        if p_param_dict["Host Star Name"] == "Qatar-6":
          p_param_dict[param] = 0.01
      print(f"\nIn the planetary parameters from the NASA Exoplanet Archive, \n\"{param}\" is listed as {val}.\n\n**** This might make EXOTIC crash. ****\n\nIf the parameter is *not* changed below, please edit it\nin the inits file before running EXOTIC.\n")
  p_param_string = json.dumps(p_param_dict)

  planetary_params = "\"planetary_parameters\": {\n"
  num_done, num_total = 0, len(p_param_dict.keys())
  for key, value in p_param_dict.items():
    num_done += 1
    if key == "Target Star RA" or key == "Target Star Dec" or key == "Planet Name" or key == "Host Star Name":
      planetary_params = planetary_params + str(f"    \"{key}\": \"{value}\",\n")
    else:
      if num_done < num_total:
        planetary_params = planetary_params + str(f"    \"{key}\": {value},\n")
      else:
        planetary_params = planetary_params + str(f"    \"{key}\": {value}\n")
  planetary_params = planetary_params + "}"

  return(planetary_params)

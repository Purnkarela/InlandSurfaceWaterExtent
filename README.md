# InlandSurfaceWaterExtent
This project processes Sentinel-2 images to calculate the Normalized Difference Water Index (NDWI) and analyze the water extent over a specified area of interest (AOI). The code includes outlier detection in the time series data and plots the water extent over time as well as the probability distribution of water extent.

## Requirements

- Python 3.7+
- Geopandas
- Rasterio
- Matplotlib
- Seaborn
- Scipy
- Satsearch
- Boto3
- Certifi
- Pyproj
- Tqdm

## Installation

To install the required packages, you can use `pip`:

```bash
pip install geopandas rasterio matplotlib seaborn scipy satsearch boto3 certifi pyproj tqdm
```

## Setup

1. **AWS Credentials**: Ensure you have AWS credentials configured for accessing Sentinel-2 data.
2. **GeoJSON File**: Prepare a GeoJSON file named `aoi.geojson` containing the AOI polygons with a field named `Name`.

## Usage

1. **Set up Environment Variables**: Ensure the SSL certificates and AWS session are properly set up.
2. Place your `aoi.geojson` file in the same directory as `mndwi_processing.py`.
3. Update the path in the script where the output images will be saved:

   ```python
   output_path = "/home/purna/Desktop/Dvara/mlai/NatureDots/images"
   ```

4. Run the script:

   ```bash
   python mndwi_processing.py
   ```

### Example

```python
import geopandas as gpd
import os
from datetime import datetime
import satsearch
from scipy.ndimage import zoom
from tqdm import tqdm
import rasterio
import certifi
import boto3
from botocore.client import Config
import sys
from pyproj import Transformer
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the main function
def MNDWI_Images(gdf, main_path, mndwi_threshold=0):
    # Code for processing images...
    pass

# Define additional helper functions
def getSubset(geotiff_file, bbox):
    # Code for getting image subset...
    pass

def validate_image(scl):
    # Code for validating image...
    pass

def compute_MNDWI(green, swir1):
    # Code for computing MNDWI...
    pass

# Set environment variables for SSL certificates and AWS session
os.environ['REQUESTS_CA_BUNDLE'] = os.path.join(os.path.dirname(sys.argv[0]), 'cacert.pem')
logging.info(certifi.where())
os.environ['REQUESTS_CA_BUNDLE'] = os.path.join(os.path.dirname(sys.argv[0]), certifi.where())
os.environ['CURL_CA_BUNDLE'] = str(certifi.where())
aws_session = rasterio.session.AWSSession(boto3.Session(), requester_pays=True)

# Read geojson file and process images
rdf = gpd.read_file("aoi.geojson")
rdf = rdf[['Name', 'geometry']]
all_water_extent_data = MNDWI_Images(rdf, "/path/to/output/directory", mndwi_threshold=0.1)

# Print or save the collected water extent data
print(all_water_extent_data)
```

## Script Details

### Main Functions

- **MNDWI_Images(gdf, main_path, mndwi_threshold=0)**:
  Processes Sentinel-2 imagery to compute NDWI for specified geographical areas within the given date range. It saves water extent images and time series plots for each area.

- **getSubset(geotiff_file, bbox)**:
  Extracts a subset of the GeoTIFF image based on the bounding box.

- **validate_image(scl)**:
  Validates the image based on the Scene Classification Layer (SCL) to filter out cloudy or otherwise invalid images.

- **compute_MNDWI(green, swir1)**:
  Computes the NDWI using the green and NIR bands from the Sentinel-2 imagery.

## Outputs

The script generates the following outputs for each AOI:

1. **Water Extent Time Series Plot**: A line chart showing the water extent over time.
2. **Water Extent Images**: MNDWI images with water extent highlighted for each date.

## Outlier Detection

The script includes functionality to detect and handle outliers in the water extent time series data using Z-scores. Outliers are replaced with the mean value of the series.

## Logging

Logging is set up to provide detailed information about the processing steps, including:
- Processing ID and time range
- Summary of Sentinel-2 items
- SCL unique values
- Invalid pixel ratio

## Contact

For any questions or issues, please contact [purnrajkarela@gmail.com].

---

This README file provides an overview of the project, setup instructions, usage examples, and details about the outputs and functionality.

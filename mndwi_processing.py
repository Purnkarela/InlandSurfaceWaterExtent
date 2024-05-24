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

def MNDWI_Images(gdf, main_path, mndwi_threshold=0):
    os.makedirs(main_path, exist_ok=True)
    
    all_water_extent_data = {}
    
    for id in gdf.Name:
        water_extent_data = {}
        id_path = os.path.join(main_path, id)
        os.makedirs(id_path, exist_ok=True)

        bbox = gdf[gdf["Name"] == id].reset_index(drop=True).total_bounds.tolist()
        start_date = datetime.strptime("2024-01-01", '%Y-%m-%d').strftime('%Y-%m-%dT00:00:00Z')
        end_date = datetime.strptime("2024-05-22", '%Y-%m-%d').strftime('%Y-%m-%dT23:59:59Z')
        timeRange = f'{start_date}/{end_date}'
        logging.info(f'Processing ID: {id} for time range: {timeRange}')
        
        SentinelSearch = satsearch.Search.search(
            url="https://earth-search.aws.element84.com/v1",
            bbox=bbox,
            datetime=timeRange,
            collections=['sentinel-2-l2a'],
            limit=1000
        )
        
        Sentinel_items = SentinelSearch.items()
        logging.info(Sentinel_items.summary(['date', 'id', 'eo:cloud_cover']))
        
        # Find the image with the least cloud cover for each unique date
        date_cloud_dict = {}
        
        for item in Sentinel_items:
            date = item.properties['datetime'][0:10]
            cloud_cover = item.properties['eo:cloud_cover']
            if cloud_cover > 10:
                continue
            if date not in date_cloud_dict or cloud_cover < date_cloud_dict[date]['cloud_cover']:
                date_cloud_dict[date] = {
                    'cloud_cover': cloud_cover,
                    'item': item
                }
        
        for date, value in tqdm(date_cloud_dict.items()):
            item = value['item']
            green_s3 = item.assets['green']['href']  # Green band
            swir1_s3 = item.assets['nir']['href']  # SWIR1 band
            scl_s3 = item.assets['scl']['href']    # Scene classification map
            
            try:
                scl, kwargs = getSubset(scl_s3, bbox)
                logging.info(f"SCL unique values: {np.unique(scl)}")  # Debugging SCL values
                
                green, kwargs = getSubset(green_s3, bbox)
                swir1, kwargs = getSubset(swir1_s3, bbox)
                
                if not validate_image(scl):
                    logging.info(f'Image on {date} is not valid due to cloud or other artifacts.')
                    continue

            except Exception as e:
                logging.error(f'Error processing field id {id}: {e}')
                continue

            mndwi = compute_MNDWI(green, swir1)
            
            # Compute water extent using the threshold value
            water_extent = np.sum(mndwi > mndwi_threshold)
            
            water_extent_data[date] = water_extent

            # Save the water extent images
            plt.imshow(mndwi, cmap='Blues', extent=bbox)
            plt.colorbar(label='MNDWI')
            plt.title(f'Water Extent on {date}')
            plt.savefig(f'{id_path}/{date}_{id}_water_extent.png')
            plt.close()
            
            del scl, mndwi, green, swir1
        
        # Save water extent data for this ID
        all_water_extent_data[id] = water_extent_data
        
        # Plot time series line chart for water extent for this ID
        dates = sorted(water_extent_data.keys())
        water_extents = [water_extent_data[date] for date in dates]
        print(water_extents)
        
        # Identify and handle outliers
        z_threshold = 2.5  # Z-score threshold for outlier detection
        z_scores = np.abs(stats.zscore(water_extents))
        
        non_outliers = np.where(z_scores <= z_threshold)[0]
        
        dates = [dates[i] for i in non_outliers]
        water_extents = [water_extents[i] for i in non_outliers]
            
        plt.figure(figsize=(10, 6))
        plt.plot(dates, water_extents, marker='o', linestyle='-')
        plt.xlabel('Date')
        plt.ylabel('Water Extent (in pixels)')
        plt.title(f'Water Extent Time Series for ID {id}')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(id_path, f'water_extent_time_series_{id}.png'))
        plt.close()

       
    # Return all water extent data
    return all_water_extent_data

def getSubset(geotiff_file, bbox):
    with rasterio.Env(aws_session):
        with rasterio.open(geotiff_file) as geo_fp:
            Transf = Transformer.from_crs("epsg:4326", geo_fp.crs) 
            lat_north, lon_west = Transf.transform(bbox[3], bbox[0])
            lat_south, lon_east = Transf.transform(bbox[1], bbox[2]) 
            x_top, y_top = geo_fp.index(lat_north, lon_west)
            x_bottom, y_bottom = geo_fp.index(lat_south, lon_east)
            window = rasterio.windows.Window.from_slices((x_top, x_bottom), (y_top, y_bottom))
            subset = geo_fp.read(1, window=window)
            kwargs = geo_fp.meta.copy()
            kwargs.update({
                'height': window.height,
                'width': window.width,
                'transform': rasterio.windows.transform(window, geo_fp.transform),
                'dtype': 'uint16'  # Update data type as per band data type
            })
    return subset, kwargs

def validate_image(scl):
    """
    Validate the image based on the scene classification layer (SCL).
    Returns True if the image is good, False otherwise.
    """
    # Define the valid classes for SCL (e.g., excluding clouds, shadows, etc.)
    valid_classes = [4, 5, 6, 7, 11]  # Example: vegetation, bare soil, water, unclassified, snow/ice
    
    invalid_pixels = np.isin(scl, valid_classes, invert=True).sum()
    total_pixels = scl.size
    invalid_ratio = invalid_pixels / total_pixels

    logging.info(f'Invalid pixel ratio: {invalid_ratio:.2%}')
    
    # Set a threshold for the acceptable ratio of invalid pixels
    invalid_threshold = 0.2  # 20% of invalid pixels
    return invalid_ratio <= invalid_threshold

def compute_MNDWI(green, swir1):
    # Resize the SWIR1 band array to match the dimensions of the green band array
    swir1_resized = swir1  # zoom(swir1, np.array(green.shape) / np.array(swir1.shape), order=1)
    
    # Compute MNDWI
    with np.errstate(divide='ignore', invalid='ignore'):
        mndwi = np.where((green + swir1_resized) != 0,
                         (green.astype(float) - swir1_resized.astype(float)) / (green + swir1_resized),
                         0)
    return mndwi

# Set environment variables for SSL certificates and AWS session
os.environ['REQUESTS_CA_BUNDLE'] = os.path.join(os.path.dirname(sys.argv[0]), 'cacert.pem')
logging.info(certifi.where())
os.environ['REQUESTS_CA_BUNDLE'] = os.path.join(os.path.dirname(sys.argv[0]), certifi.where())
os.environ['CURL_CA_BUNDLE'] = str(certifi.where())
aws_session = rasterio.session.AWSSession(boto3.Session(), requester_pays=True)

# Read geojson file and process images
rdf = gpd.read_file("aoi.geojson")
rdf = rdf[['Name', 'geometry']]
all_water_extent_data = MNDWI_Images(rdf, "/home/purna/Desktop/Dvara/mlai/NatureDots/images", mndwi_threshold=0.1)

# Print or save the collected water extent data
print(all_water_extent_data)

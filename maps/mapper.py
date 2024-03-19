import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import folium

def get_exif_data(image):
    """Return a dictionary from the exif data of an PIL Image item. Also converts the GPS Tags"""
    exif_data = {}
    info = image._getexif()
    if info:
        for tag, value in info.items():
            decoded = TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                gps_data = {}
                for gps_tag in value:
                    sub_decoded = GPSTAGS.get(gps_tag, gps_tag)
                    gps_data[sub_decoded] = value[gps_tag]
                exif_data[decoded] = gps_data
            else:
                exif_data[decoded] = value
    return exif_data

def get_lat_lon(exif_data):
    """Returns the latitude and longitude, if present, from the provided exif_data"""
    lat = None
    lon = None

    if "GPSInfo" in exif_data:
        gps_info = exif_data["GPSInfo"]

        if "GPSLatitude" in gps_info and "GPSLatitudeRef" in gps_info:
            e = gps_info["GPSLatitude"]
            lat = (e[0][0] / e[0][1] +
                   e[1][0] / e[1][1] / 60 +
                   e[2][0] / e[2][1] / 3600
                  ) * (-1 if gps_info["GPSLatitudeRef"] == 'S' else 1)

        if "GPSLongitude" in gps_info and "GPSLongitudeRef" in gps_info:
            e = gps_info["GPSLongitude"]
            lon = (e[0][0] / e[0][1] +
                   e[1][0] / e[1][1] / 60 +
                   e[2][0] / e[2][1] / 3600
                  ) * (-1 if gps_info["GPSLongitudeRef"] == 'W' else 1)

    return lat, lon

def plot_on_map(locations):
    """Plots the given locations on a map"""
    map = folium.Map(location=[60, -95], zoom_start=3)
    for lat, lon in locations:
        if lat and lon:
            folium.Marker([lat, lon]).add_to(map)
    return map

def main(directory_path):
    locations = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg"):
                image_path = os.path.join(root, file)
                image = Image.open(image_path)
                exif_data = get_exif_data(image)
                lat, lon = get_lat_lon(exif_data)
                if lat and lon:
                    locations.append((lat, lon))

    map = plot_on_map(locations)
    map.save('map.html')

if __name__ == "__main__":
    directory_path = 'path/to/your/images'
    main(directory_path)

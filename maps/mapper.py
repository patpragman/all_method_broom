import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import folium
from statistics import mean
from datetime import datetime
from copy import deepcopy

"""
glorious chatgpt4 generated code that I used to produced to make a pretty folium map, I had to debug the extraction
process quite a bit, and come up with a way to extract the timestamps too - but that was fairly easy.
"""


def get_exif_data(image):
    """Return a dictionary from the exif data of an PIL Image item. Also converts the GPS Tags"""
    exif_data = {}
    info = image._getexif()

    if info:
        exif_data["GPSInfo"] = {}
        for tag, value in info.items():
            decoded = TAGS.get(tag, tag)

            if decoded == "GPSInfo":
                gps_data = {}
                for gps_tag in value:
                    sub_decoded = GPSTAGS.get(gps_tag, gps_tag)
                    gps_data[sub_decoded] = value[gps_tag]
                exif_data[decoded] = gps_data
            elif decoded == 'DateTime':
                dt = datetime.strptime(info[tag], "%Y:%m:%d %H:%M:%S")
                exif_data["GPSInfo"]["timestamp"] = int(datetime.timestamp(dt))
            else:
                exif_data[decoded] = value
    return exif_data


def get_lat_lon(exif_data):
    """Returns the latitude and longitude, if present, from the provided exif_data"""
    lat = None
    lon = None

    if all(key in exif_data["GPSInfo"] for key in ["timestamp", "GPSLatitude", "GPSLongitude"]):
        # extract the data from the GPS reference data
        gps_info = exif_data["GPSInfo"]
        latsign = -1 if gps_info["GPSLatitudeRef"] == 'S' else 1
        lonsign = -1 if gps_info["GPSLongitudeRef"] == 'W' else 1
        lat = latsign * sum([val / (60 ** i) for i, val in enumerate(gps_info['GPSLatitude'])])
        lon = lonsign * sum([val / (60 ** i) for i, val in enumerate(gps_info['GPSLongitude'])])

        return float(lat), float(lon), exif_data["GPSInfo"]['timestamp']
    else:
        return 0, 0, 0


def plot_on_map(locations):
    """Plots the given locations on a map as a path"""
    mean_lat = mean([lat for lat, lon, ts in locations])
    mean_lon = mean([lon for lat, lon, ts in locations])
    map = folium.Map(location=[mean_lat, mean_lon], zoom_start=3)

    # now that we have the locations and their timestamps, iterate through, and find the differences

    tracks = []
    working_track = []
    i = 0
    spacetime_tuples = [tup for tup in locations]
    while len(spacetime_tuples) > 2:

        (lat1, lon1, t1), (lat2, lon2, t2) = spacetime_tuples.pop(0), spacetime_tuples.pop(1)
        diff = t2 - t1

        if diff > 10:
            tracks.append(deepcopy(working_track))
            working_track = []
            spacetime_tuples.insert(0, (lat2, lon2, t2))
            continue
        else:
            working_track.append((lat1, lon1))
            spacetime_tuples.insert(0, (lat2, lon2, t2))

    for i, track in enumerate(tracks):
        if track:
            fg = folium.FeatureGroup(name=f'Track {i + 1}')
            folium.PolyLine(track, color="red", weight=2.5, opacity=1).add_to(fg)
            fg.add_to(map)

    # Add a LayerControl to toggle tracks
    folium.LayerControl().add_to(map)

    return map


def main(directory_path):
    locations = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(".jpg") or file.lower().endswith(".jpeg"):
                image_path = os.path.join(root, file)
                image = Image.open(image_path)
                exif_data = get_exif_data(image)
                lat, lon, timestamp = get_lat_lon(exif_data)
                if lat and lon:
                    locations.append((lat, lon, timestamp))

    locations.sort(key=lambda x: x[2])
    map = plot_on_map(locations)
    map.save('map.html')


if __name__ == "__main__":
    directory_path = '/home/patrickpragman/PycharmProjects/adventures_in_elodea/Pics2022'
    main(directory_path)

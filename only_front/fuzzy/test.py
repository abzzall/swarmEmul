import osmium
import osm2geojson
import pyproj
import xml.etree.ElementTree as ET

def osm_to_geojson(osm_file):
    class OSMHandler(osmium.SimpleHandler):
        def area(self, a):
            if 'building' in a.tags:
                self.areas.append(a)

    handler = OSMHandler()
    handler.areas = []
    handler.apply_file(osm_file)

    features = osm2geojson.jsonio.as_features(handler.areas)
    feature_collection = osm2geojson.jsonio.as_feature_collection(features)

    return feature_collection

def create_gazebo_world_description(geojson_data):
    root = ET.Element("world", name="default")

    # Set the world properties
    physics = ET.SubElement(root, "physics", name="default", type="ode")
    ET.SubElement(physics, "gravity").text = "0 0 -9.81"

    scene = ET.SubElement(root, "scene")
    ET.SubElement(scene, "ambient").text = "0.4 0.4 0.4 1"
    ET.SubElement(scene, "background").text = "0.7 0.7 0.7 1"

    # Define the buildings as obstacles
    for feature in geojson_data['features']:
        if feature['geometry']['type'] == 'Polygon':
            coordinates = feature['geometry']['coordinates'][0]

            # Calculate the centroid and use it as the obstacle position
            centroid = [sum(x) / len(coordinates) for x in zip(*coordinates)]
            x, y = centroid

            # Use the average building height as the obstacle height
            height = 5.0  # You can replace this with the actual height extracted from OSM data

            # Create a simple box-shaped obstacle for each building
            obstacle = ET.SubElement(root, "model", name=f"building_{x}_{y}")
            ET.SubElement(obstacle, "pose").text = f"{x} {y} {height / 2} 0 0 0"
            ET.SubElement(obstacle, "link", name="link")
            collision = ET.SubElement(ET.SubElement(obstacle, "link", name="link"), "collision", name="collision")
            ET.SubElement(collision, "geometry")
            ET.SubElement(collision.find("geometry"), "box", size=f"1 1 {height}")

    tree = ET.ElementTree(root)
    return tree

def save_gazebo_world_file(tree, output_file):
    tree.write(output_file, xml_declaration=True, encoding='utf-8')

def main():
    osm_file = "path/to/your/file.osm"
    geojson_data = osm_to_geojson(osm_file)
    gazebo_world_file = "path/to/output/file.world"

    gazebo_world_description = create_gazebo_world_description(geojson_data)
    save_gazebo_world_file(gazebo_world_description, gazebo_world_file)

if __name__ == "__main__":
    main()

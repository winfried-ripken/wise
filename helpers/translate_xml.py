import xml.etree.ElementTree as ET


# Insert <Parameters> part only
from helpers.visual_parameter_def import watercolor_vp_ranges


def parse_xml_definition():
    tree = ET.parse("glsnippet.txt")
    for p in tree.getroot():
        if p.get("type") == "float":
            start = p.find("minrange")
            stop = p.find("maxrange")

            print(f'("{p.get("id")}", {start.text}, {stop.text}),')


# Insert <preset> part only
def parse_xml_preset(vp_ranges=None):
    tree = ET.parse("glsnippet.txt")
    for p in tree.getroot()[0]:
        if vp_ranges is None or p.get("ref") in [x[0] for x in vp_ranges]:
            print(f'("{p.get("ref")}", {p.text}),')


if __name__ == '__main__':
    parse_xml_preset(watercolor_vp_ranges)

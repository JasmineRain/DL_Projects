import xmltodict
import dicom

def getROI(file):
    xml_file = open(file, 'r')
    xml_str = xml_file.read()
    xml_dict = xmltodict.parse(xml_str)
    array = xml_dict['plist']['dict']['array']['dict']['array']['dict']
    calcs = []
    for item in array:
        for calc in item['array'][1]['string']:
            values = calc.replace(' ', '').replace('(', '').replace(')', '').split(',')
            calcs += [[int(float(x)) for x in values]]
    return calcs


ds = dicom.read_file("20587054_b6a4f750c6df4f90_MG_R_CC_ANON.dcm")
print(ds.dir())

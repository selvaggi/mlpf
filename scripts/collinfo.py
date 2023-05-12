
import uproot
import argparse

default_input='/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/p8_ee_ZZ_ecm240/events_057786065.root'

parser = argparse.ArgumentParser(description='Args')
parser.add_argument('-i', '--input', default=default_input)
args = parser.parse_args()

infile = uproot.open(args.input)
if 'podio_metadata;1' in infile.keys():
    podio_table = file['podio_metadata']['events___idTable']['m_names'].array()[0]
elif 'metadata;1' in infile.keys():
    podio_table = infile['metadata']['CollectionIDs']['m_names'].array()[0]
else:
    print('ERROR: File format not known! Aborting...')
    exit(1)

print("")
print("ID  ->  Collection")
print("==================")
for i,val in enumerate(podio_table):
    print("{:2}  ->  {}".format(i+1, val))
print("==================")
print("")

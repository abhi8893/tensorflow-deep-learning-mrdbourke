''' Download Medical Cost Data'''

import requests
import os

OUTDIR = 'data/medical_cost'
OUTNAME = 'medical_cost.csv'

url = r'https://gist.githubusercontent.com/meperezcuello/82a9f1c1c473d6585e750ad2e3c05a41/raw/d42d226d0dd64e7f5395a0eec1b9190a10edbc03/Medical_Cost.csv'
resp = requests.get(url)
data = resp.content.decode('utf-8')


if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)


outfile = os.path.join(OUTDIR, OUTNAME)

with open(outfile, 'w') as f:
    f.write(data)
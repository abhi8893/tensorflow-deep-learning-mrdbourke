import wget
import zipfile
import os
import shutil

DATA_URL = r'https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_10_percent.zip'
OUT_FILE = r'data/10_food_classes_10_percent.zip'

out_file = wget.download(DATA_URL, out=OUT_FILE)

with zipfile.ZipFile(out_file, 'r') as zipref:
    zipref.extractall('data/')

os.remove(out_file)
shutil.rmtree('data/__MACOSX')
import wget
import zipfile
import os
import shutil

DATA_URL = r"https://storage.googleapis.com/ztm_tf_course/nlp_getting_started.zip"
OUT_FILE = r'data/disaster_tweets.zip'

out_file = wget.download(DATA_URL, out=OUT_FILE)
os.makedirs('data/disaster_tweets')

with zipfile.ZipFile(out_file, 'r') as zipref:
    zipref.extractall('data/disaster_tweets')


os.remove(out_file)
import os
import glob
from PIL import Image

os.makedirs('images', exist_ok=True)
files = glob.glob('jpg/*.jpg')
for f in files:
    img = Image.open(f)
    img = img.resize((64, 64))
    target = os.path.join('images', os.path.basename(f))
    img.save(target)

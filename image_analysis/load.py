import os
import os.path
from urllib.request import urlopen
from image_analysis.image import Image

DATA_FOLDER = 'data'

FILENAME_TEMPLATE = '{}.jpg'
URL_TEMPLATE = 'http://drive.google.com/uc?export=view&id={}'

def load_image(id):
    filename = FILENAME_TEMPLATE.format(id)
    full_path = os.path.join(os.getcwd(), DATA_FOLDER, filename)
    url = URL_TEMPLATE.format(id)

    if not os.path.exists(full_path):
        with open(full_path, 'wb') as f:
            with urlopen(url) as img:
                img_data = img.read()
                f.write(img_data)

    img = Image(id, full_path)

    os.remove(full_path)

    return img

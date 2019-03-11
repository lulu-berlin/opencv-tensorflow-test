from flask import Flask
import json
from image_analysis import load_image

app = Flask(__name__)

image_ids = [
    '1qwQNKQrUox-m3-5d5XbUgpqj7_45IDf7',
    '1oLOj9jLdcZeYXGl5hMlcc6gw1_R_c1Ma',
    '1bxIqp1RDSbTDmHkvVvkpZJ73cyRhGdyi',
    '1lGr6QonQgxT91GLyCpXCNW2fWLv9QZL2',
    '1YXjyJfXwPDW_VPcVivWND-ZZctFmq9oT',
    '1qwQNKQrUox-m3-5d5XbUgpqj7_45IDf7'
]

@app.route("/")
def images():
    images = [load_image(id) for id in image_ids]

    most_yellowish = max(images, key=lambda img: img.yellowishness()).id

    return json.dumps({
        'images': {
            img.id: {
                'shape': list(img.shape()),
                'averageGrayscaleIntensity': img.average_grayscale_intensity(),
                'objects': img.objects()
                }
            for img in images
            },
        'mostYellowish': most_yellowish
    })

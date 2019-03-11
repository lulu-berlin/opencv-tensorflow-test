import cv2
import numpy as np
import tensorflow as tf

YELLOW_MIN = np.array([25, 146, 190], np.uint8)
YELLOW_MAX = np.array([62, 174, 250], np.uint8)

MODEL_FILE = 'ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb'


class Image:
    graph_def = None
    
    def __init__(self, id, path):
        self.id = id
        self.img = cv2.imread(path, cv2.IMREAD_COLOR)

        if Image.graph_def is None:
            with tf.gfile.GFile(MODEL_FILE, 'rb') as f:
                Image.graph_def = tf.GraphDef()
                Image.graph_def.ParseFromString(f.read())

    def shape(self):
        return self.img.shape

    def average_grayscale_intensity(self):
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        flat = np.ndarray.flatten(gray_img)
        return np.average(flat)

    def yellowishness(self):
        yellow_mask = cv2.inRange(self.img, YELLOW_MIN, YELLOW_MAX)
        yellow_count = cv2.countNonZero(yellow_mask)
        total_pixels = self.img.shape[0] * self.img.shape[1]
        return yellow_count / total_pixels

    def objects(self):
        result = []

        with tf.Session() as sess:
            # Restore session
            sess.graph.as_default()
            tf.import_graph_def(Image.graph_def, name='')

            # Read and preprocess an image.
            rows = self.img.shape[0]
            cols = self.img.shape[1]
            inp = cv2.resize(self.img, (300, 300))
            inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

            # Run the model
            out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                            sess.graph.get_tensor_by_name(
                                'detection_scores:0'),
                            sess.graph.get_tensor_by_name('detection_boxes:0'),
                            sess.graph.get_tensor_by_name('detection_classes:0')],
                           feed_dict={'image_tensor:0': inp.reshape(1,
                                                                    inp.shape[0],
                                                                    inp.shape[1],
                                                                    3)})

            # Visualize detected bounding boxes.
            num_detections = int(out[0][0])
            for i in range(num_detections):
                classId = int(out[3][0][i])
                score = float(out[1][0][i])
                bbox = [float(v) for v in out[2][0][i]]
                if score > 0.3:
                    x = bbox[1] * cols
                    y = bbox[0] * rows
                    right = bbox[3] * cols
                    bottom = bbox[2] * rows

                    result.append({
                        'x': x,
                        'y': y,
                        'right': right,
                        'bottom': bottom,
                        'classId': classId
                    })

        return result

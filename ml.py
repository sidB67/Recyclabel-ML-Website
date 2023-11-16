from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

def predict(img):
    model = tf.keras.models.load_model('recycleimageclassifierMobileDense64.h5')
    data = np.ndarray(shape=(1, 256, 256, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (256, 256)
    image = ImageOps.fit(image, size)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 255.0) 

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    return prediction
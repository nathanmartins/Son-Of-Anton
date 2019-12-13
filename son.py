import logging
import math
import os
import pickle
import re

import PIL.Image
import numpy as np
from mtcnn import MTCNN
from numpy import expand_dims
from sklearn import preprocessing, neighbors
from tensorflow_core.python.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
# DEBUG = True
DEBUG = False

model = load_model('facenet_keras.h5')

logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)


def extract_faces(img_path: str):
    faces_arr = list()

    # Open file and convert to numpy
    image_array = np.array(PIL.Image.open(img_path).convert("RGB"), "uint8")

    detector = MTCNN()
    faces = detector.detect_faces(image_array)

    if len(faces) == 0:
        # If there are no people in a training image, skip the image.
        logging.warning(f"Image {img_path} not suitable for training. Size{len(faces)}")
        return None, None

    for face in faces:
        logging.debug(f"Image {img_path} is suitable for training!")

        x1, y1, width, height = face['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = image_array[y1:y2, x1:x2]

        # resize pixels to the model size
        image = PIL.Image.fromarray(face)
        image = image.resize((160, 160))
        faces_arr.append(np.asarray(image))

    return faces_arr, faces


def get_embedding(face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    return model.predict(samples)


def prepare():
    x_train = list()
    y_labels = list()

    # Loop through each person in the training set
    for label in os.listdir(TRAIN_DIR):

        path = os.path.join(TRAIN_DIR, label)

        # This will ignore anything that is not jpg|jpeg|png *USE WITH CAUTION*
        allowed_files = [os.path.join(path, f) for f in os.listdir(path) if
                         re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]

        for img_path in allowed_files:

            logging.debug(f"File: {img_path}, Label: {label}")

            faces, _ = extract_faces(img_path)
            if faces is not None:
                for face in faces:
                    x_train.append(np.asarray(face))
                    y_labels.append(label)

    # Converting string labels into numbers.
    le = preprocessing.LabelEncoder()
    labels_encoded = le.fit_transform(y_labels)

    with open("x_train.pickle", 'wb') as f:
        pickle.dump(x_train, f)

    with open("y_labels.pickle", 'wb') as f:
        pickle.dump(y_labels, f)

    with open("labels_encoded.pickle", 'wb') as f:
        pickle.dump(labels_encoded, f)


def train():
    with open("x_train.pickle", 'rb') as f:
        x_train = pickle.load(f)
        # x_train = np.array(x_train)
        # x_train = np.reshape(x_train, (-1, 2))

    with open("labels_encoded.pickle", 'rb') as f:
        y_labels = pickle.load(f)

    # convert each face in the train set to an embedding
    encoded_x_train = list()
    for face_pixels in x_train:
        embedding = get_embedding(face_pixels)[0]
        encoded_x_train.append(embedding)
    encoded_x_train = np.asarray(encoded_x_train)

    # Determine how many neighbors to use for weighting in the KNN classifier.
    n_neighbors = int(round(math.sqrt(len(x_train))))
    logging.info(f"n_neighbors: {n_neighbors}")

    # Create and train the KNN classifier.
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm="ball_tree", weights='distance')
    knn_clf.fit(encoded_x_train, y_labels)

    # Save the trained KNN classifier
    with open("model.clf", 'wb') as f:
        pickle.dump(knn_clf, f)


def predict():
    with open("model.clf", 'rb') as f:
        knn_clf = pickle.load(f)

    with open("labels_encoded.pickle", 'rb') as f:
        y_labels = pickle.load(f)

    le = preprocessing.LabelEncoder()
    le.fit_transform(y_labels)

    for img in os.listdir(TEST_DIR):

        full_path = os.path.join(TEST_DIR, img)

        faces, raws = extract_faces(full_path)

        if faces is None:
            logging.warning(f"WARNING: COULD NOT FIND A FACE IN {full_path}")
            continue

        c = 0

        for face in faces:

            faces_encodings = get_embedding(face)

            # A list of tuples of found face locations in css (top, right, bottom, left) order
            x_face_locations = tuple(raws[c]["box"])
            c += 1

            # Use the KNN model to find the best matches for the test face
            closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
            are_matches = list()

            for i in range(len(x_face_locations)):
                try:

                    dis = closest_distances[0][i][0]

                    logging.debug(f"Closest distance is  {dis} - {dis <= 6}")

                    if dis < 7:
                        logging.debug(f"Adding a Dis {dis}")
                        are_matches.append(dis)
                except IndexError:
                    pass

            logging.debug(f"Dis is {are_matches}")

            pred = knn_clf.predict(faces_encodings)

            for pred, loc, rec in zip(pred, x_face_locations, are_matches):

                if rec:
                    if pred == 1:
                        a = "unknown"
                    else:
                        a = "nsm"
                    logging.info(f"Found: {a} - {img}")
                else:
                    logging.info("unknown")


if __name__ == '__main__':
    # prepare()
    # train()
    predict()

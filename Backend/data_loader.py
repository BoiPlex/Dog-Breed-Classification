import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

BASEPATH = "/Users/chinghaochang/project-170/data/images/"  # image's path

def load_and_preprocess_image(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# use "train_test_split" to random split load_data function 
def load_data_with_split():
    LABELS = set()
    paths = []
    for d in os.listdir(BASEPATH):
        LABELS.add(d)
        paths.append((BASEPATH + d, d))
    
    X, y = [], []
    for path, label in paths:
        for image_path in os.listdir(path):
            image = load_and_preprocess_image(os.path.join(path, image_path))
            X.append(image)
            y.append(label)
    
    encoder = LabelBinarizer()
    X = np.array(X)
    y = encoder.fit_transform(np.array(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    return X_train, X_test, y_train, y_test, encoder, len(LABELS)

# use .mat to split load_data function
def load_data_from_mat():
    import scipy.io
    mat_data = scipy.io.loadmat("/Users/chinghaochang/project-170/data/lists/test_list.mat")

    file_list = [item[0][0] for item in mat_data['file_list']]
    labels = [int(label[0]) for label in mat_data['labels']]

    images = []
    for file_path in file_list:
        full_path = os.path.join(BASEPATH, file_path)
        image = cv2.imread(full_path)
        if image is not None:
            image = cv2.resize(image, (224, 224))
            images.append(image)
        else:
            print(f"Warning: Image at {full_path} could not be loaded.")

    X = np.array(images)
    y = np.array(labels)

    encoder = LabelBinarizer()
    y = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    return X_train, X_test, y_train, y_test, encoder, len(encoder.classes_)

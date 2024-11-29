import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import scipy.io

BASEPATH = "/Users/chinghaochang/project-170/data/Images/"

def load_data_from_mat(mat_file_path):
    
    mat_data = scipy.io.loadmat(mat_file_path)

    
    if "file_list" not in mat_data or "labels" not in mat_data:
        raise ValueError("The .mat file must contain 'file_list' and 'labels'.")

    file_list = [item[0] for item in mat_data['file_list'].flatten()]
    labels = mat_data['labels'].flatten()

    X = []
    for file_path in file_list:
        full_path = os.path.join(BASEPATH, file_path)
        img = load_img(full_path, target_size=(224, 224))
        X.append(img_to_array(img))
    X = np.array(X, dtype="float32") / 255.0

    encoder = LabelBinarizer()
    y = encoder.fit_transform(labels - 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    return X_train, X_test, y_train, y_test, encoder, len(encoder.classes_)

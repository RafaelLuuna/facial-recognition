import os
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


from sklearn.svm import SVC

from .loader import FaceLoader
from .utils import extract_face


class FaceDetector:
    def __init__(self, directory, label_encoder_path=None, model_path=None):
        self.loader = FaceLoader(directory)
        print('[ LOADER SET ]')

        self.label_encoder = LabelEncoder()
        self.model_svc = SVC(kernel='linear', probability=True)

        file_path = os.path.dirname(os.path.realpath(__file__))

        self.label_encoder_path  = label_encoder_path or os.path.join(file_path,'models','label_encoder.pkl')
        self.model_path  = model_path or os.path.join(file_path,'models','model_svc.pkl')

    def embedd_faces(self):
        EMBEDDED_X = []
        for embedding in self.loader.X:
            EMBEDDED_X.append(embedding)

        self.EMBEDDED_X = EMBEDDED_X

    def encript_labels(self):
        Y = self.loader.Y
        self.label_encoder.fit(Y)
        self.ENCRIPTED_Y = self.label_encoder.transform(Y)

    
    def train(self):
        self.encript_labels()
        self.embedd_faces()
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.EMBEDDED_X, self.ENCRIPTED_Y, shuffle=True, random_state=17)
        self.model_svc.fit(self.train_x, self.train_y)


    def predict(self, imgDirectory):
        img, landmarks = extract_face(imgDirectory, include_landmarks=True)
        embeddings = landmarks['embedding']
        embeddings = np.expand_dims(embeddings, axis=0)
        ypred = self.model_svc.predict(embeddings)

        return ypred
    

    def save_model(self):
        joblib.dump(self.model_svc, self.model_path)
        joblib.dump(self.label_encoder, self.label_encoder_path)
        print(f'[ MODEL SAVED to {self.model_path} ]')

    def load_model(self):
        if os.path.exists(self.label_encoder_path) and os.path.exists(self.model_path):
            self.model_svc = joblib.load(self.model_path)
            self.label_encoder = joblib.load(self.label_encoder_path)
            print(f'[ MODEL LOADED from {self.model_path} ]')
        else:
            print('[ MODEL NOT FOUND. PLEASE TRAIN FIRST ]')


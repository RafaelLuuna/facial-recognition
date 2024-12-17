from .face_loader import FaceLoader
import numpy as np

import numpy as np
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


class FaceDetector:
    def __init__(self, directory):
        self.loader = FaceLoader(directory)
        print('[ LOADER SET ]')
        self.embedder = FaceNet()
        self.lbEncoder = LabelEncoder()
        self.svcModel = SVC(kernel='linear', probability=True)

    def _get_embedding(self, img):
        img = img.astype('float32')
        img = np.expand_dims(img, axis=0)

        yhat = self.embedder.embeddings(img)

        return yhat[0]

    def embedd_faces(self):
        EMBEDDED_X = []
        for face in self.loader.X:
            EMBEDDED_X.append(self._get_embedding(face))

        self.EMBEDDED_X = EMBEDDED_X

    def encript_labels(self):
        Y = self.loader.Y
        self.lbEncoder.fit(Y)
        self.ENCRIPTED_Y = self.lbEncoder.transform(Y)

    
    def train(self):
        self.encript_labels()
        self.embedd_faces()
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.EMBEDDED_X, self.ENCRIPTED_Y, shuffle=True, random_state=17)
        self.svcModel.fit(self.train_x, self.train_y)


    def predict(self, imgDirectory):
        img = self.loader.extract_face(imgDirectory)
        img = img.astype('float32')
        img = self._get_embedding(img)
        img = np.expand_dims(img, axis=0)
        ypred = self.svcModel.predict(img)

        return ypred


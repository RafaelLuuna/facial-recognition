import numpy as np
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


class FaceProcessor:
    def __init__(self, loader):
        self.loader = loader
        self.embedder = FaceNet()
        self.lbEncoder = LabelEncoder()
        self.model = SVC(kernel='linear', probability=True)

    def get_embedding(self, face_img):
        face_img = face_img.astype('float32')
        face_img = np.expand_dims(face_img, axis=0)

        self.EMBEDDED_X = self.embedder.embeddings(face_img)

    def encript_labels(self):
        self.lbEncoder.fit(self.)
        self.ENCRIPTED_Y = self.lbEncoder.transform(Y)

        return Y
    
    def train(self):
        Y = self.encript_labels(self.loader.Y)
        EMBEDDED_X = self.get_embedding(self.loader.X)

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.EMBEDDED_X, self.ENCRIPTED_Y, shuffle=True, random_state=17)

        self.model.fit(self.train_x, self.train_y)


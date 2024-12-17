import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from mtcnn.mtcnn import MTCNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class FaceLoader:
  def __init__(self, directory):
    self.directory = directory
    self.target_size = (160,160)
    self.X = []
    self.Y = []
    self.mtcnnModel = MTCNN()

  def extract_face(self, dir):
    img = cv.imread(dir)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = self.mtcnnModel.detect_faces(img)
    x,y,w,h = results[0]['box']
    x,y = abs(x), abs(y)
    face = img[y:y+h , x:x+w]
    face_arr = cv.resize(face, self.target_size)
    return face_arr

  def load_faces(self, dir):
    FACES = []
    for fname in os.listdir(dir):
      try:
        path = os.path.join(dir, fname)
        FACES.append(self.extract_face(path))
      except:
        pass
    return FACES

  def load_classes(self):
    for sub_dir in os.listdir(self.directory):
      path = os.path.join(self.directory, sub_dir,'')
      FACES = self.load_faces(path)
      LABELS = [sub_dir for _ in range(len(FACES))]
      self.X.extend(FACES)
      self.Y.extend(LABELS)
    return np.asarray(self.X), np.asarray(self.Y)

  def plot_images(self):
    for num, img in enumerate(self.X):
      ncols = 3
      nrows = len(self.Y)//ncols
      plt.subplot(nrows, ncols, num+1)
      plt.imshow(img)
      plt.axis('off')


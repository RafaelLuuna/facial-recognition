import os
import numpy as np
from .utils import extract_face

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class FaceLoader:
  def __init__(self, directory):
    self.directory = directory
    self.X = []
    self.Y = []

  def load_faces(self, dir):
    FACES = []
    for fname in os.listdir(dir):
      try:
        path = os.path.join(dir, fname)
        FACES.append(extract_face(path))
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


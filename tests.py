import os
import joblib
import time
import numpy as np
import cv2 as cv
from pprint import pprint
from face_detector.utils import categorize_faces, save_images, extract_face, extract_faces_from_dir, split_array
from face_detector.detector import FaceDetector
from keras_facenet import FaceNet


inicio_ecxec = time.time()


faces, landmarks = extract_face('/home/user/projects/facial-detection/test/fotos-EAI/original/2022-12-21 - EAÍ Fim de ano - 0151.jpg',include_landmarks=True)

save_images(face,'/home/user/projects/facial-detection/test/fotos-EAI/resized')
quit()
# faces_dir = '/home/user/projects/facial-detection/test/fotos-EAI/original'
# save_dir = '/home/user/projects/facial-detection/test/fotos-EAI/classes'



# # FACES, LANDMARKS = extract_faces_from_dir(faces_dir, include_landmarks=True, limit=10)

# FACES = joblib.load('/home/user/projects/facial-detection/test/cache/FACES.pkl')
# LANDMARKS = joblib.load('/home/user/projects/facial-detection/test/cache/LANDMARKS.pkl')
# # FACES = split_array(FACES,500)

# # model = FaceNet()
# # EMBEDDINGS = model.embeddings(FACES)

# KEYPOINTS = [landmark['keypoints'] for landmark in LANDMARKS]
# EMBEDDINGS = [[
#     lm['keypoints']['nose'][0], lm['keypoints']['nose'][1],
#     lm['keypoints']['mouth_left'][0], lm['keypoints']['mouth_left'][1],
#     lm['keypoints']['mouth_right'][0], lm['keypoints']['mouth_right'][1],
#     lm['keypoints']['left_eye'][0], lm['keypoints']['left_eye'][1],
#     lm['keypoints']['right_eye'][0], lm['keypoints']['right_eye'][1]
#     # *lm['embedding']
# ] for lm in LANDMARKS]

# FACES_CATEGORIZED = categorize_faces(EMBEDDINGS)

# labels = FACES_CATEGORIZED.labels_

# print(f'\nSalvando arquivos clusterizados...')
# for label in np.unique(labels):
#     # if label != -1:  # -1 indica outliers
#     path = os.path.join(save_dir, f'cluster_{label}')
#     os.makedirs(path, exist_ok=True)

# for i, label in enumerate(labels):
#     # if label != -1: # -1 indica outliers
#     path = os.path.join(save_dir, f'cluster_{label}',f'{i}.jpg')
#     img = FACES[i]
#     cv.imwrite(path, img)

# print(f'\nImagens salvas com sucesso!')


# fim_exec=time.time()
# temp_exec=fim_exec-inicio_ecxec
# print(f'\n\nTempo de execução: {temp_exec}\nInício: {inicio_ecxec}\nFim: {fim_exec}')
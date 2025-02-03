import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import joblib
import math

import face_recognition
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet

from deepface import DeepFace

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

TARGET_SIZE = (200,200)

def redim_img(img, max_height):
    x_size, y_size = img.shape[:2]
    ratio = x_size / y_size
    new_x = int(x_size*ratio)
    new_img = cv.resize(img, (new_x,max_height))
    return new_img


def extract_face(dir, target_size=TARGET_SIZE, include_landmarks=False):
    img = cv.imread(dir)
    img = img.astype('float32')
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = redim_img(img,1600)

    model = FaceNet()
    landmarks = model.extract(img, threshold=0.95)
    
    if include_landmarks == True:
        
    else:
        model = MTCNN()
        landmarks = model.detect_faces(img)

    face_arr = []

    for landmark in landmarks:
        x,y,w,h = landmark['box']
        x,y = abs(x), abs(y)
        face_img = img[y:y+h , x:x+w]


        # face = cv.resize(face_img, target_size)
        face = cv.cvtColor(face_img, cv.COLOR_BGR2RGB)

        face_arr.append(face)

    
    res = face_arr

    if include_landmarks == True:
        res = (face_arr, landmarks)
    return res

def extract_faces_from_dir(img_dir, save_dir='', target_size=TARGET_SIZE, include_landmarks=False, limit=0):
    FACES = []
    LANDMARKS = []


    for i, img in enumerate(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img)
        if os.path.exists(img_path):
            print(f'\nProcurando rostos em: {img_path}...')

            if include_landmarks:
                faces, landmarks = extract_face(img_path, target_size=target_size, include_landmarks=True)
                LANDMARKS.extend(landmarks)
            else:
                faces = extract_face(img_path, target_size=target_size, include_landmarks=False)

            FACES.extend(faces)


            print(f'\n{len(faces)} rostos encontrados.')
            

            #--- Salva uma cópia da variável FACES e LANDMARKS caso precise carregar futuramente. ---#
            try:
                os.remove( '/home/user/projects/facial-detection/test/cache/FACES.pkl')
                os.remove( '/home/user/projects/facial-detection/test/cache/LANDMARKS.pkl')

            except FileNotFoundError:
                pass    

            joblib.dump(FACES, '/home/user/projects/facial-detection/test/cache/FACES.pkl')
            joblib.dump(LANDMARKS, '/home/user/projects/facial-detection/test/cache/LANDMARKS.pkl')
            #----------------------------------------------------------------------------------------#

            if save_dir != '':
                save_images(faces,save_dir)
            
            if limit > 0 and i >= limit:
                break
    return (FACES, LANDMARKS) if include_landmarks else FACES


def save_images(img_list, save_path):
    print('\nSalvando arquivos...')
    try:
        if save_path != '':
            for img in img_list:
                i = len(os.listdir(save_path))
                path = os.path.join(save_path, f'{i}.jpg')
                cv.imwrite(path, img)
        print('Arquivos salvos com sucesso!')
    except Exception as e:
       print(f'Erro ao salvar arquivos: {e}')

def categorize_faces(embeddings):

    embeddings = np.array(embeddings)

    print('Normalizando rostos...')
    normalizer = StandardScaler()
    embeddings_scaled = normalizer.fit_transform(embeddings)
    print('operação concluída.')

    print('Clusterizando imagens...')
    cluster = DBSCAN(eps=0.2, min_samples=2)
    cluster.fit(embeddings_scaled)
    print('operação concluída')
    return cluster


def split_array(arr, limit):
    res = []
    pages = math.ceil(len(arr) / limit)
    for page in range(pages):
        i = page*limit
        sub_array = arr[i:i+limit]
        res.append(sub_array)
    return res

        
    



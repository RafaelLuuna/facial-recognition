from face_detector import FaceDetector

detector = FaceDetector('./dataset')
detector.loader.load_classes()

detector.train()

predict = detector.predict('/home/user/projects/facial-detection/dataset/sardor_abdirayimov/8.jpg')

print(detector.lbEncoder.inverse_transform(predict))


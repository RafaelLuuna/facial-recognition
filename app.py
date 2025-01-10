from face_detector.detector import FaceDetector
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os



detector = FaceDetector('./dataset')

if os.path.exists(detector.model_path) and os.path.exists(detector.label_encoder_path):
    detector.load_model()
else:
    detector.loader.load_classes()
    detector.train()
    detector.save_model()


app = Flask(__name__)
@app.route('/upload_file', methods=['GET','POST'])
def upload_files():
    if request.method == 'POST':
        try:
            file = request.files['file']
            filename = secure_filename(file.filename)
            path = f'static/uploads/cache/{filename}'
            file.save(path)

            result = detector.predict(path)
            result = detector.label_encoder.inverse_transform(result)

            os.remove(path)
            return f'Este rosto é do(a): {result}'
        
        except KeyError as e:
            print(f'\n\nVariável não encontrada na requisição POST: {e}\n')
        
        except Exception as e:
            print(f'\n\nErro ao carregar arquivo: {e}\n')


    if request.method == 'GET':
        try:
            return render_template('reconhecimento_facial/upload.html')
        except Exception as e:
            print(f'\n\nErro ao carregar formulário: {e}\n')
    
    
    return f'Erro ao receber requisição {request.method}'




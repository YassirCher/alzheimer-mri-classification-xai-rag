from flask import Flask, render_template, request, jsonify, send_file
import torch
import numpy as np
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import json

from config import config
from models.inference import get_model
from utils.gradcam import CustomGradCAM, overlay_cam_on_image, denormalize_image
from utils.rag_gemini import get_rag

app = Flask(__name__)
app.config.from_object(config['development'])

# Create folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('data', exist_ok=True)

def load_metrics():
    """Load metrics from JSON file"""
    metrics_file = os.path.join(os.path.dirname(__file__), 'data', 'model_metrics.json')
    try:
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading metrics: {e}")
    return {
        'overall_accuracy': 0.9998,
        'macro_f1': 0.9998,
        'weighted_f1': 0.9998,
        'cohen_kappa': 0.9997,
        'macro_auc': 1.0000,
        'per_class_metrics': {
            'MildDemented': {'precision': 1.0000, 'recall': 1.0000, 'f1': 1.0000, 'support': 1019, 'auc': 1.0000},
            'ModerateDemented': {'precision': 1.0000, 'recall': 1.0000, 'f1': 1.0000, 'support': 1004, 'auc': 1.0000},
            'NonDemented': {'precision': 1.0000, 'recall': 0.9992, 'f1': 0.9996, 'support': 1298, 'auc': 1.0000},
            'VeryMildDemented': {'precision': 0.9991, 'recall': 1.0000, 'f1': 0.9995, 'support': 1079, 'auc': 1.0000}
        },
        'confusion_matrix': [[1019, 0, 0, 0], [0, 1004, 0, 0], [0, 0, 1297, 1], [0, 0, 0, 1079]]
    }

metrics = load_metrics()

@app.route('/')
def index():
    """Home dashboard"""
    stats = {
        'accuracy': f"{metrics['overall_accuracy']*100:.2f}%",
        'macro_f1': f"{metrics['macro_f1']:.4f}",
        'cohen_kappa': f"{metrics['cohen_kappa']:.4f}",
        'total_test_samples': 4400
    }
    return render_template('index.html', stats=stats, metrics=metrics)

# Updated /predict route that supports GET for UI and POST for API prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        """Prediction endpoint"""
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type'}), 400
            
            img = Image.open(io.BytesIO(file.read())).convert('RGB')
            model = get_model()
            result = model.predict(img)
            
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            
            return jsonify({
                'success': True,
                'class_name': result['class_name'],
                'confidence': result['confidence'],
                'probabilities': result['probabilities'],
                'image': f"data:image/png;base64,{img_base64}"
            })
        
        except Exception as e:
            print(f"Error: {e}")
            return jsonify({'error': str(e)}), 500

    else:
        # GET request: Serve the predict.html page with upload UI
        return render_template('predict.html')

@app.route('/api/gradcam', methods=['POST'])
def generate_gradcam():
    """Generate GradCAM for specific class"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file'}), 400
        
        file = request.files['file']
        class_idx = int(request.form.get('class_idx', 0))
        
        # Read and process image
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # Get model and make prediction
        model = get_model()
        prediction_result = model.predict(img)
        
        # Generate GradCAM
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        target_layer = model.model.features[-1][0]
        gradcam = CustomGradCAM(model.model, target_layer, device)
        
        img_tensor = model.transform(img).unsqueeze(0).to(device)
        img_tensor_grad = img_tensor.clone().detach().requires_grad_(True)
        
        cam = gradcam.generate_cam(img_tensor_grad, class_idx)
        
        img_np = denormalize_image(img_tensor.detach())
        overlay, heatmap, cam_resized = overlay_cam_on_image(img_np, cam, alpha=0.6)
        
        def encode_img(arr):
            fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
            if len(arr.shape) == 2:
                ax.imshow(arr, cmap='jet')
            else:
                ax.imshow(arr)
            ax.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            plt.close(fig)
            return base64.b64encode(buf.getvalue()).decode()
        
        # Encode original image
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'original': f"data:image/png;base64,{img_base64}",
            'heatmap': f"data:image/png;base64,{encode_img(cam_resized)}",
            'overlay': f"data:image/png;base64,{encode_img(overlay)}",
            'class_name': prediction_result['class_name'],
            'confidence': prediction_result['confidence']
        })
    
    except Exception as e:
        print(f"GradCAM Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(error):
    return render_template('500.html'), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint for RAG chatbot"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Get RAG response
        rag = get_rag()
        result = rag.query(user_message)
        
        # Clean up the response
        response_text = result['answer']
        
        return jsonify({
            'success': True,
            'response': response_text,
            'sources': [s['source'] for s in result['sources']]
        })
    
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/chat-page')
def chat_page():
    """Dedicated chat page"""
    return render_template('chat.html')

@app.route('/xai-analysis')
def xai_analysis():
    """XAI Analysis page"""
    return render_template('xai_analysis.html')

if __name__ == '__main__':
    print("🧠 Starting Alzheimer's AI Dashboard...")
    print(f"📊 Accuracy: {metrics['overall_accuracy']*100:.2f}%")
    print("🌐 Open http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)

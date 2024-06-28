from flask import Flask, request, render_template, redirect, url_for
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load pre-trained feature extractor
class FeatureExtractor(nn.Module):
    def __init__(self, pretmodel):
        super(FeatureExtractor, self).__init__()
        self.model = pretmodel
        self.model = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x):
        with torch.no_grad():
            features = self.model(x)
        return features.view(features.size(0), -1)

# Simple Fully Connected Network
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1152)
        self.bn1 = nn.BatchNorm1d(1152)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1152, 768)
        self.bn2 = nn.BatchNorm1d(768)
        self.fc3 = nn.Linear(768, output_dim)
        self.bn3 = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)
        return x

# Initialize networks
feature_extractor = FeatureExtractor(models.efficientnet_b3(pretrained=True))
feature_dim = 1536  # EfficientNet-B3 output feature dimension
model = SimpleNN(feature_dim, 2)

# Load model checkpoint
model.load_state_dict(torch.load('models/classifier_final.pth.tar', map_location=torch.device('cuda')))

# Set models to evaluation mode
feature_extractor.eval()
model.eval()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Prediction function
def predict_image(image_path, model):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)
    
    features = feature_extractor(image)
    output = model(features)
    
    _, predicted = torch.max(output, 1)
    return "Real" if predicted.item() == 0 else "Fake"

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            prediction = predict_image(filepath, model)
            return render_template('result.html', prediction=prediction, image_path=filepath)
    
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static' ,filename='uploads/' + filename))

if __name__ == '__main__':
    app.run(debug=True)
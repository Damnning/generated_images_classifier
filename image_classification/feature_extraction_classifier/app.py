import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
import numpy as np

app = Flask(__name__)

# Define the model architecture (same as in training script)
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # Remove the last layer

    def forward(self, x):
        with torch.no_grad():
            features = self.model(x)
        return features.view(features.size(0), -1)

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize networks
feature_extractor = FeatureExtractor()
feature_dim = 2048  # ResNet50 output feature dimension
teacher = SimpleNN(feature_dim, 2)
student = SimpleNN(feature_dim, 2)
binary_classifier = BinaryClassifier(2)

# Load trained weights
teacher.load_state_dict(torch.load('models/teacher_10ep.pth', map_location=torch.device('cpu')))
student.load_state_dict(torch.load('models/student_10ep.pth', map_location=torch.device('cpu')))
binary_classifier.load_state_dict(torch.load('models/classifier_10ep.pth', map_location=torch.device('cpu')))

# Set models to evaluation mode
teacher.eval()
student.eval()
binary_classifier.eval()

# Define transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Extract features
    features = feature_extractor(image)

    # Get teacher and student outputs
    with torch.no_grad():
        teacher_outputs = teacher(features)
        student_outputs = student(features)

    # Calculate discrepancy
    discrepancy = (teacher_outputs - student_outputs) ** 2

    # Get prediction
    with torch.no_grad():
        outputs = binary_classifier(discrepancy)
        _, predicted = torch.max(outputs, 1)
    
    return 'Real' if predicted.item() == 0 else 'Fake'

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            prediction = predict(file_path)
            return render_template('result.html', prediction=prediction)
    return render_template('upload.html')

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
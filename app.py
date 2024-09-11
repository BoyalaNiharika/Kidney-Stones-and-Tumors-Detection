import sqlite3
import datetime
from flask import Flask, render_template, request, redirect, url_for
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import io
import base64
import os
import cv2
import numpy as np
from torchvision.models import resnet50

app = Flask(__name__, static_url_path='/static')


# Function to create SQLite database and tables
def create_tables():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY, username TEXT, password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS user_history
                 (id INTEGER PRIMARY KEY, username TEXT, date TEXT, predicted_label_alexnet TEXT)''')
    conn.commit()
    conn.close()

# Function to register new user
def register_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
    conn.commit()
    conn.close()

# Function to verify user credentials during login
def verify_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    user = c.fetchone()
    conn.close()
    return user

# Function to insert user history into database
def insert_user_history(username, date, predicted_label_alexnet):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("INSERT INTO user_history (username, date, predicted_label_alexnet) VALUES (?, ?, ?)",
              (username, date, predicted_label_alexnet))
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = verify_user(username, password)
        if user:
            return redirect(url_for('user_page'))
        else:
            return render_template('login.html', message='Invalid username or password.')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        register_user(username, password)
        return redirect(url_for('login'))
    return render_template('register.html')

num_classes_alexnet = 3
alexnet_model = models.alexnet(pretrained=False, num_classes=num_classes_alexnet)
alexnet_model.features[0] = nn.Sequential(alexnet_model.features[0], nn.BatchNorm2d(64))
alexnet_model.features[3] = nn.Sequential(alexnet_model.features[3], nn.BatchNorm2d(192))
alexnet_model.features[6] = nn.Sequential(alexnet_model.features[6], nn.BatchNorm2d(384))
alexnet_model.features[8] = nn.Sequential(alexnet_model.features[8], nn.BatchNorm2d(256))
alexnet_model.features[10] = nn.Sequential(alexnet_model.features[10], nn.BatchNorm2d(256))
alexnet_model.classifier[-1] = nn.Linear(4096, num_classes_alexnet)

alexnet_model.load_state_dict(torch.load('kidney_stone_detection_alexnet_model (1).pth', map_location=torch.device('cpu')))
alexnet_model.eval()

# Image preprocessing for AlexNet
preprocess_alexnet = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the trained ResNet model
class FeatureHook:
    def __init__(self, model, layer_name):
        self.features = None
        self.hook = model.layer4[-1].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

resnet_model_path = 'kidney_stone_detection_resnet_model.pth'
resnet_model = resnet50(pretrained=False)
num_classes_resnet = 3  # Change this based on the number of classes in your dataset
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes_resnet)
resnet_model.load_state_dict(torch.load(resnet_model_path, map_location=torch.device('cpu')))
resnet_model.eval()

# Image preprocessing for ResNet
preprocess_resnet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_image_alexnet(image_path):
    # Preprocess the input image
    input_image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess_alexnet(input_image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    # Make the prediction using AlexNet
    with torch.no_grad():
        alexnet_model.eval()
        output = alexnet_model(input_batch)

    # Get the predicted class
    predicted_class_alexnet = torch.argmax(F.softmax(output[0], dim=0)).item()

    # Map the predicted class index to the actual class label
    class_labels_alexnet = ['Normal', 'Stone', 'Tumor']
    predicted_label_alexnet = class_labels_alexnet[predicted_class_alexnet]

    return predicted_label_alexnet

def predict_image(image_path):
    # Load the trained model with map_location specified
    model_path = 'kidney_stone_detection_resnet_model.pth'
    model = resnet50(pretrained=False)
    num_classes = 3  # Change this based on the number of classes in your dataset
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Load the model with map_location=torch.device('cpu')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Define the image transformation
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load the input image
    input_image = Image.open(image_path).convert("RGB")

    # Preprocess the input image
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    # Check if GPU is available and move the model and input tensor to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_batch = input_batch.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Make the prediction
    with torch.no_grad():
        output = model(input_batch)

    # Get the predicted class index
    predicted_class_index = torch.argmax(output[0]).item()

    # Map the predicted class index to the corresponding class label
    predicted_class_label = ["Normal", "Tumor", "Stone"][predicted_class_index]

    # Create an instance of FeatureHook for the last convolutional layer of layer4
    hook = FeatureHook(model, "layer4")

    # Forward pass to get the feature map
    output = model(input_batch)
    feature_map = hook.features
    hook.hook.remove()  # Remove the hook

    # Calculate Grad-CAM
    grads = torch.autograd.grad(output[0, predicted_class_index], feature_map)[0]
    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
    grad_cam = torch.sum(weights * feature_map, dim=1, keepdim=True)

    # Upsample the Grad-CAM heatmap to the input image size
    grad_cam_upsampled = torch.nn.functional.interpolate(grad_cam, size=(input_image.height, input_image.width),
                                                         mode="bilinear", align_corners=False)

    # Normalize Grad-CAM values to range [0, 1]
    grad_cam_np = (grad_cam_upsampled - grad_cam_upsampled.min()) / (grad_cam_upsampled.max() - grad_cam_upsampled.min())
    grad_cam_np = grad_cam_np.detach().cpu().numpy().squeeze(0).squeeze(0)


    # Convert images to base64 for embedding in HTML
    input_image_base64 = image_to_base64(input_image)
    grad_cam_base64 = image_to_base64(Image.fromarray((grad_cam_np * 255).astype(np.uint8)))

    # Create superimposed image
    superimposed_img = create_superimposed_image(input_image, grad_cam_np)

    # Convert superimposed image to base64
    superimposed_img_base64 = image_to_base64(Image.fromarray(superimposed_img))

    # Save Grad-CAM and superimposed images
    save_path_grad_cam = 'static/grad_cam.jpg'
    save_path_superimposed = 'static/superimposed_image.jpg'

    save_image(grad_cam_base64, save_path_grad_cam)
    save_image(superimposed_img_base64, save_path_superimposed)

    return  input_image_base64, grad_cam_base64, superimposed_img_base64

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def save_image(base64_string, save_path):
    img_data = base64.b64decode(base64_string)
    with open(save_path, 'wb') as f:
        f.write(img_data)

def create_superimposed_image(input_image, grad_cam_np):
    # Normalize Grad-CAM values to range [0, 1]
    grad_cam_normalized = (grad_cam_np - grad_cam_np.min()) / (grad_cam_np.max() - grad_cam_np.min())

    # Create a heatmap with color intensity based on Grad-CAM values
    heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam_normalized), cv2.COLORMAP_JET)

    # Convert heatmap to RGBA
    heatmap_rgba = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGBA)

    # Convert input image to RGBA and then to NumPy array
    input_image_rgba = input_image.convert("RGBA")
    input_image_np = np.array(input_image_rgba)

    # Convert superimposed image to NumPy array
    superimposed_img_np = cv2.addWeighted(input_image_np, 0.5, heatmap_rgba, 0.5, 0)

    # Convert superimposed image back to RGBA
    superimposed_img_rgba = cv2.cvtColor(superimposed_img_np, cv2.COLOR_RGBA2RGB)

    return superimposed_img_rgba

@app.route('/user', methods=['GET', 'POST'])
def user_page():
    predicted_label_alexnet = None
    input_image_base64_alexnet = None
    grad_cam_base64_alexnet = None
    superimposed_img_base64_alexnet = None
    history_data = None

    if request.method == 'POST':
            file = request.files['file']

            if file:
                file_path = 'static/uploaded_image.jpg'
                file.save(file_path)

                (predicted_label_alexnet) = predict_image_alexnet(file_path)
                (input_image_base64_alexnet, grad_cam_base64_alexnet, 
                superimposed_img_base64_alexnet) = predict_image(file_path)

                username = 'test_user'
                date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                insert_user_history(username, date, predicted_label_alexnet)



    return render_template('user_page.html',
                           predicted_label_alexnet=predicted_label_alexnet,
                           input_image_base64_alexnet=input_image_base64_alexnet,
                           grad_cam_base64_alexnet=grad_cam_base64_alexnet,
                           superimposed_img_base64_alexnet=superimposed_img_base64_alexnet,
                           history_data=history_data)

@app.route('/user_history', methods=['GET'])
def show_user_history():
    # Fetch user history from the database
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT date, predicted_label_alexnet FROM user_history WHERE username = ?", ('test_user',))  # Replace 'test_user' with actual username
    history_data = c.fetchall()
    conn.close()

    return render_template('user_history.html', history_data=history_data)



if __name__ == '__main__':
    create_tables()  # Create SQLite table on application start
    app.run(debug=True)

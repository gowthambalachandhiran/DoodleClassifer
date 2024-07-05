import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# Define transformations (same as training)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),  # Resize to 224x224 as specified
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])  # Using ImageNet normalization
])

def load_model():
    model = models.resnet18(pretrained=True)  # Load the pre-trained ResNet-18 model
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 250)

    device = torch.device('cpu')
    model.to(device)
    model.eval()
    
    return model, device

def load_labels(filepath):
    with open(filepath, 'r') as f:
        labels = f.read().splitlines()
    return labels

def predict_image(model, device, image, labels):
    try:
        image = Image.open(io.BytesIO(image)).convert('L')  # Convert image to grayscale
    except Exception as e:
        raise e

    image = transform(image).unsqueeze(0).to(device)  # Apply transformations and add batch dimension

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, preds = torch.max(probabilities, 1)
        predicted_class = preds.item()
        confidence_score = confidence.item()

    predicted_label = labels[predicted_class]
    return predicted_label, confidence_score

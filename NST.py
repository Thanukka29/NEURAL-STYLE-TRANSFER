# Upload images: dhoni.jpg (content) and beach.jpg (style)
from google.colab import files
uploaded = files.upload()  # Choose the two images when prompted
# Import required libraries
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load and preprocess image
def load_image(path, size=400):
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)
# Convert tensor to displayable image
def show_image(tensor, title="Image"):
    image = tensor.cpu().clone().squeeze(0)
    image = image.permute(1, 2, 0).clamp(0, 1)
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()
# Gram matrix for style
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    return torch.mm(features, features.t()) / (c * h * w)

# Load content and style images
content = load_image("dhoni.jpg")
style = load_image("beach.jpg", size=content.shape[-1])

# Display original images
show_image(content, "Content: Dhoni")
show_image(style, "Style: Beach")

# Load VGG19 model
vgg = models.vgg19(pretrained=True).features.to(device).eval()
for param in vgg.parameters():
    param.requires_grad = False
# Layers to extract features from
content_layer = '21'
style_layers = ['0', '5', '10', '19', '28']
# Extract features
def get_features(image, model):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in style_layers + [content_layer]:
            features[name] = x
    return features
# Get features
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)
style_grams = {l: gram_matrix(style_features[l]) for l in style_layers}

# Prepare target image
target = content.clone().requires_grad_(True)
optimizer = torch.optim.Adam([target], lr=0.003)

# Optimization loop
for step in range(201):
    target_features = get_features(target, vgg)
    # Content loss
    content_loss = torch.mean((target_features[content_layer] - content_features[content_layer])**2)
    # Style loss
    style_loss = 0
    for l in style_layers:
        target_gram = gram_matrix(target_features[l])
        style_gram = style_grams[l]
        style_loss += torch.mean((target_gram - style_gram)**2)
    # Total loss
    total_loss = 1e4 * content_loss + 1e2 * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    if step % 50 == 0:
        print(f"Step {step}, Loss: {total_loss.item():.2f}")
# Show final result
show_image(target, "Stylized Output: Dhoni + Beach Style")

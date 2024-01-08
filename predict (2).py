import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    learning_rate = checkpoint['learning_rate']
    epochs = checkpoint['epochs']
    optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

    return model

def process_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image)

    return image

def predict(image_path, model, topk=5, gpu=False):
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    image = process_image(image_path)
    image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        output = model(image)

    probabilities, classes = torch.topk(F.softmax(output, dim=1), topk)
    probabilities = probabilities.squeeze().tolist()
    classes = [model.class_to_idx[idx] for idx in classes.squeeze().tolist()]

    return probabilities, classes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', action='store', default='checkpoint.pth')
    parser.add_argument('--top_k', dest='top_k', default='3')
    parser.add_argument('--filepath', dest='filepath', default='flowers/test/1/image_06743.jpg') # use a deafault filepath to a primrose image 
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store', default='gpu')

    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint)

    if model is not None:
        probabilities, classes = predict(args.image_path, model, args.top_k, args.gpu)

        print("Predictions:")
        for prob, cls in zip(probabilities, classes):
            print(f"Class: {cls}, Probability: {prob:.4f}")

if __name__ == "__main__":
    main()

from argparse import ArgumentParser
from torchvision import transforms
from PIL import Image
from net import Net 
import numpy as np
import torch
import cv2


#argument parsing
#getting directory with images
parser = ArgumentParser(description="import images")
parser.add_argument("-i", "--images", 
                    nargs = "?",
                    required=True,
                    help = "path to images dir")
args = parser.parse_args()

frame = cv2.imread(args.images)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (224, 224))
edges = cv2.Canny(gray, 64, 64)

pimage = Image.fromarray(np.uint8(edges)).convert('L')


model = Net()
model.load_state_dict(torch.load("model.pt"))
model.eval()

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
#   transforms.Normalize((0.5), (0.5))
])

img_n = transform(pimage)
img_n = img_n.unsqueeze(0)

prediction = model(img_n)
print(prediction)

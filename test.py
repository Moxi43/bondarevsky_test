from PIL import Image
from torchvision import transforms
from net import Net
import torch

model = Net()
model.load_state_dict(torch.load("model.pt"))
model.eval()

img = Image.open("data/doing/1.jpg")

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
#    transforms.Normalize((0.5), (0.5))
    ])

img_n = transform(img)
img_n = img_n.unsqueeze(0)

prediction = model(img_n)
print(prediction)

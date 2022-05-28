import pandas as pd
import torch, torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from ctran import ctranspath


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
trnsfrms_val = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std)
    ]
)
class roi_dataset(Dataset):
    def __init__(self, img_csv,
                 ):
        super().__init__()
        self.transform = trnsfrms_val

        self.images_lst = img_csv

    def __len__(self):
        return len(self.images_lst)

    def __getitem__(self, idx):
        path = self.images_lst.filename[idx]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)


        return image

img_csv=pd.read_csv(r'./test_list.csv')
test_datat=roi_dataset(img_csv)
database_loader = torch.utils.data.DataLoader(test_datat, batch_size=1, shuffle=False)

model = ctranspath()
model.head = nn.Identity()
td = torch.load(r'./ctranspath.pth')
model.load_state_dict(td['model'], strict=True)


model.eval()
with torch.no_grad():
    for batch in database_loader:
        features = model(batch)
        features = features.cpu().numpy()


#

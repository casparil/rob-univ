from univ.utils.folder2lmdb import ImageFolderLMDB
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import argparse
import pandas as pd
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("lmdb_path")
parser.add_argument("output_path", help="Needs to end in .csv")
args = parser.parse_args()
assert args.output_path.endswith(".csv")

data = ImageFolderLMDB(
    args.lmdb_path,
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
)
loader = DataLoader(data, batch_size=1024, shuffle=False, num_workers=2)

labels = []
for _, y_batch in tqdm(loader):
    labels.append(y_batch)
labels = torch.concat(labels).numpy()
series = pd.Series(labels)
print("First 3 labels")
print(series.head(3))
print("Last 3 labels")
print(series.tail(3))
series.to_csv(args.output_path)

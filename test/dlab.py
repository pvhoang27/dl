from torchvision.datasets import VOCSegmentation
import numpy as np

dataset = VOCSegmentation(root = "my_pascal_voc", year = "2012", image_set = "trainval")

image , label = dataset[1200]

print(np.unique(np.array(label)))
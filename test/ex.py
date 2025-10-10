from PIL import Image
import numpy as np

image = Image.open("2007_000241.png")
image = np.array(image)
print(image.shape)
print(np.unique(image))
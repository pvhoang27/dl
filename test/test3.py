import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.transforms import ToTensor
from PIL import Image

# Nạp model
model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, num_classes=21)  # Số class tùy bạn
checkpoint = torch.load("trained_models/early_stop.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Đọc ảnh và transform
img = Image.open("2007_000032.jpg").convert("RGB")
img_tensor = ToTensor()(img).unsqueeze(0)  # Thêm batch dimension

# Dự đoán
with torch.no_grad():
    output = model(img_tensor)

print(output)  # Xem kết quả: boxes, labels, scores
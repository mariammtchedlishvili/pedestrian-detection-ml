import os
import time
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw
import gradio as gr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_detector(weights_path="best_pedestrian_detector.pth", num_classes=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded trained weights from {weights_path}")
    else:
        print("Trained weights file not found locally; initialized model structure.")

    model.to(device)
    model.eval()
    return model

model = load_detector()

@torch.no_grad()
def detect_pedestrians(input_img, conf_threshold=0.5):
    if input_img is None:
        return None, "გთხოვთ ატვირთოთ სურათი."

    pil_img = input_img.convert("RGB")
    img_tensor = F.to_tensor(pil_img).to(device)

    t0 = time.time()
    output = model([img_tensor])[0]
    elapsed_ms = (time.time() - t0) * 1000

    keep = output["scores"] >= conf_threshold
    boxes = output["boxes"][keep].cpu().numpy()
    scores = output["scores"][keep].cpu().numpy()

    draw = ImageDraw.Draw(pil_img)
    details = []
    for i, (box, score) in enumerate(zip(boxes, scores), start=1):
        xmin, ymin, xmax, ymax = box
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        draw.text((xmin + 4, max(ymin - 15, 2)), f"Person {score:.2f}", fill="red")
        details.append(
            f"ქვეითი #{i}: სიზუსტე {score*100:.1f}% | ჩარჩო: [{int(xmin)}, {int(ymin)}, {int(xmax)}, {int(ymax)}]"
        )

    summary = f"ნაპოვნია {len(boxes)} ქვეითი ({elapsed_ms:.1f} ms)\n" + "\n".join(details)
    return pil_img, summary

demo = gr.Interface(
    fn=detect_pedestrians,
    inputs=[
        gr.Image(type="pil", label="ატვირთეთ ფოტო"),
        gr.Slider(minimum=0.1, maximum=0.95, value=0.5, step=0.05, label="Confidence Threshold (ზღვარი)")
    ],
    outputs=[
        gr.Image(type="pil", label="შედეგი (Detected Pedestrians)"),
        gr.Textbox(label="დეტალური ინფორმაცია")
    ],
    title="Pedestrian Detection System (Faster R-CNN)",
    description="ატვირთეთ ქუჩის სურათი ქვეითების აღმოსაჩენად და აირჩიეთ სასურველი Confidence ზღვარი."  
)

if __name__ == "__main__":
    demo.launch()

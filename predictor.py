import torch
import matplotlib
matplotlib.use("Agg")
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from torchvision import transforms
from PIL import Image
from nn_models import build_efficientnet

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Models", "best_efficientnet.pth")
SAVE_DIR   = os.path.join(BASE_DIR, "gradcam_outputs")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Load model once at startup ────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_efficientnet(model_name="b3", dropout=0.57, freeze_backbone=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
# print(f"✓ Model loaded from: {MODEL_PATH}")

# ── Preprocessing ─────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

CLASS_NAMES = ["Benign", "Malignant"]

# ── GradCAM Generator ─────────────────────────────────────────────
def _generate_gradcam(img_tensor):
    gradients  = []
    activations = []

    # Register hooks fresh every call — removed persistent startup hook
    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    target_layer = model.features[-1]
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    img_tensor = img_tensor.unsqueeze(0).to(device)

    # torch.enable_grad() ensures grads flow even inside model.eval()
    with torch.enable_grad():
        output     = model(img_tensor)
        pred_idx   = output.argmax(1).item()
        confidence = torch.softmax(output, dim=1)[0][pred_idx].item()

        model.zero_grad()
        output[0, pred_idx].backward()

    # Remove hooks immediately after use
    fh.remove()
    bh.remove()

    grad = gradients[0].cpu().numpy()[0]     # (C, H, W)
    act  = activations[0].cpu().numpy()[0]   # (C, H, W)

    weights = grad.mean(axis=(1, 2))
    cam     = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (300, 300))
    cam -= cam.min()
    if cam.max() != 0:
        cam /= cam.max()

    return cam, pred_idx, confidence

# ── Main predict function ─────────────────────────────────────────
def predict(image_path: str) -> dict:
    orig_img     = Image.open(image_path).convert("RGB")
    orig_resized = orig_img.resize((300, 300))
    orig_np      = np.array(orig_resized) / 255.0
    img_tensor   = transform(orig_img)

    cam, pred_idx, confidence = _generate_gradcam(img_tensor)
    pred_class = CLASS_NAMES[pred_idx]

    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0).to(device))
        probs  = torch.softmax(output, dim=1)[0].cpu().numpy()

    all_probs = {cls: float(probs[i]) for i, cls in enumerate(CLASS_NAMES)}

    heatmap = cm.jet(cam)[:, :, :3]
    overlay = np.clip(0.5 * orig_np + 0.5 * heatmap, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(orig_np);  axes[0].set_title("Original Image", fontsize=12);       axes[0].axis("off")
    axes[1].imshow(cam, cmap="jet"); axes[1].set_title("GradCAM Heatmap\n(red = high attention)", fontsize=12); axes[1].axis("off")

    color  = "red"   if pred_class == "Malignant" else "green"
    symbol = "⚠️"    if pred_class == "Malignant" else "✓"
    axes[2].imshow(overlay)
    axes[2].set_title(f"{symbol} Prediction: {pred_class}\nConfidence: {confidence:.2%}",
                      fontsize=12, color=color)
    axes[2].axis("off")

    plt.suptitle("Skin Lesion Analysis — EfficientNet-B3", fontsize=14, fontweight="bold")
    plt.tight_layout()

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(SAVE_DIR, f"{base_name}_{pred_class}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ GradCAM saved to: {save_path}")

    return {
        "prediction"  : pred_class,
        "confidence"  : confidence,
        "all_probs"   : all_probs,
        "gradcam_path": save_path
    }
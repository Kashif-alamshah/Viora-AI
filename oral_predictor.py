import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from torchvision import transforms
from PIL import Image
from nn_models import oral_cnn

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Models", "best_oral_cnn.pth")
SAVE_DIR   = os.path.join(BASE_DIR, "gradcam_outputs")
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = oral_cnn(conv1_out=128, conv2_out=128, conv3_out=128,
                 dropout=0.2196809136375538, neurons=128)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print(f"✓ Oral CNN model loaded from: {MODEL_PATH}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

CLASS_NAMES = ["Benign", "Malignant"]

def _generate_gradcam(img_tensor):
    gradients  = []
    activations = []

    target_layer = model.feature[-4]

    fh = target_layer.register_forward_hook(
        lambda m, i, o: activations.append(o.detach())
    )
    bh = target_layer.register_full_backward_hook(
        lambda m, gi, go: gradients.append(go[0].detach())
    )

    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.enable_grad():
        output     = model(img_tensor)
        pred_idx   = output.argmax(1).item()
        confidence = torch.softmax(output, dim=1)[0][pred_idx].item()

        model.zero_grad()
        output[0, pred_idx].backward()

    fh.remove()
    bh.remove()

    grad = gradients[0].cpu().numpy()[0]
    act  = activations[0].cpu().numpy()[0]

    weights = grad.mean(axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= cam.min()
    if cam.max() != 0:
        cam /= cam.max()

    return cam, pred_idx, confidence

def oral_predict(image_path: str) -> dict:
    orig_img     = Image.open(image_path).convert("RGB")
    orig_resized = np.array(orig_img.resize((224, 224))) / 255.0
    img_tensor   = transform(orig_img)

    cam, pred_idx, confidence = _generate_gradcam(img_tensor)
    pred_class = CLASS_NAMES[pred_idx]

    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0).to(device))
        probs  = torch.softmax(output, dim=1)[0].cpu().numpy()

    all_probs = {cls: float(probs[i]) for i, cls in enumerate(CLASS_NAMES)}

    heatmap = cm.jet(cam)[:, :, :3]
    overlay = np.clip(0.5 * orig_resized + 0.5 * heatmap, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(orig_resized); axes[0].set_title("Original");        axes[0].axis("off")
    axes[1].imshow(cam, cmap="jet"); axes[1].set_title("GradCAM Heatmap"); axes[1].axis("off")
    color = "red" if pred_class == "Malignant" else "green"
    axes[2].imshow(overlay)
    axes[2].set_title(f"Prediction: {pred_class}\nConfidence: {confidence:.2%}", color=color)
    axes[2].axis("off")

    plt.suptitle("Oral Lesion Analysis — CNN", fontsize=14, fontweight="bold")
    plt.tight_layout()

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(SAVE_DIR, f"{base_name}_oral_{pred_class}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "prediction"  : pred_class,
        "confidence"  : confidence,
        "all_probs"   : all_probs,
        "gradcam_path": save_path
    }
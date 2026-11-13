import os
import cv2
import gradio as gr
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_LABELS = {
    "akiec": "Actinic Keratosis / Bowen's Disease (akiec) [Pre-Malignant]",
    "bcc": "Basal Cell Carcinoma (bcc) [Malignant]",
    "bkl": "Benign Keratosis-like Lesion (bkl) [Benign]",
    "df": "Dermatofibroma (df) [Benign]",
    "mel": "Melanoma (mel) [Malignant]",
    "nv": "Melanocytic Nevus (nv) [Benign]",
    "vasc": "Vascular Lesion (vasc) [Benign]",
}

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

WEIGHTS_PATH = "best_model.pth" if os.path.exists("best_model.pth") else "weights/best_model.pth"
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(CLASSES), drop_rate=0.3)
if os.path.exists(WEIGHTS_PATH):
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

def predict_and_explain(input_pil: Image.Image):
    if input_pil is None:
        return {}, None, "Please upload or select a dermatoscopic image."

    rgb_pil = input_pil.convert("RGB")
    raw_resized = np.array(rgb_pil.resize((IMG_SIZE, IMG_SIZE)))
    inp_tensor = eval_tf(rgb_pil).unsqueeze(0).to(DEVICE)

    activations, gradients = [], []
    target_layer = model.conv_head
    fwd_hook = target_layer.register_forward_hook(lambda m, i, o: activations.append(o.detach()))
    bwd_hook = target_layer.register_full_backward_hook(lambda m, gi, go: gradients.append(go[0].detach()))

    try:
        model.zero_grad()
        logits = model(inp_tensor)
        probs = F.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        logits[0, pred_idx].backward()

        weights = torch.mean(gradients[0], dim=(2, 3), keepdim=True)
        cam = F.relu(torch.sum(weights * activations[0], dim=1).squeeze(0)).cpu().numpy()
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        overlay_rgb = cv2.addWeighted(raw_resized, 0.58, heatmap_rgb, 0.42, 0)
    finally:
        fwd_hook.remove()
        bwd_hook.remove()

    probs_np = probs.detach().cpu().numpy()
    top_indices = np.argsort(probs_np)[::-1][:3]
    top3_dict = {CLASS_LABELS[CLASSES[i]]: float(probs_np[i]) for i in top_indices}

    top_code = CLASSES[top_indices[0]]
    top_conf = probs_np[top_indices[0]] * 100
    summary_md = (
        f"### Primary Prediction: **{CLASS_LABELS[top_code]}** ({top_conf:.1f}%)\n"
        f"- **Grad-CAM Target Layer:** `EfficientNet-B0 conv_head` ($7 \\times 7$ spatial feature map)\n"
        f"- **Model Benchmark (HAM10000 Held-Out Test Set, $N=992$):** Macro ROC-AUC = `0.9654` | Weighted ROC-AUC = `0.9662` | Accuracy = `83.97%`\n"
        f"- *Disclaimer: Research demonstration for biomedical engineering portfolio evaluation only. Not for clinical diagnosis.*"
    )
    return top3_dict, overlay_rgb, summary_md

CANDIDATE_SAMPLES = [
    "samples/sample_mel.jpg", "assets/samples/sample_mel.jpg",
    "samples/sample_bcc.jpg", "assets/samples/sample_bcc.jpg",
    "samples/sample_nv.jpg", "assets/samples/sample_nv.jpg",
    "samples/sample_bkl.jpg", "assets/samples/sample_bkl.jpg",
]
SAMPLE_FILES = [f for f in CANDIDATE_SAMPLES if os.path.exists(f)]

with gr.Blocks(theme=gr.themes.Soft(), title="HAM10000 CNN + Grad-CAM Explainability") as demo:
    gr.Markdown(
        """
        # Automated Dermatoscopic Skin Lesion Classification with Grad-CAM Explainability
        **Architecture:** `EfficientNet-B0` (Smoothed Inverse-Frequency Class-Weighted Loss) | **Dataset:** ISIC 2018 / HAM10000 (Leakage-Free `lesion_id` Split)
        **Test Set Performance ($N=992$):** Macro ROC-AUC: **0.9654** | Weighted ROC-AUC: **0.9662** | Macro Specificity: **96.19%** | Overall Accuracy: **83.97%**
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Dermatoscopic Lesion Image")
            submit_btn = gr.Button("Run CNN Classification & Grad-CAM", variant="primary")
            if SAMPLE_FILES:
                gr.Examples(
                    examples=SAMPLE_FILES,
                    inputs=image_input,
                    label="Held-Out Test Set Samples (Melanoma, Basal Cell Carcinoma, Nevus, Benign Keratosis)",
                )
        with gr.Column(scale=1):
            label_output = gr.Label(num_top_classes=3, label="Top-3 Diagnostic Class Probabilities")
            gradcam_output = gr.Image(label="Grad-CAM Morphological Attention Overlay (JET Colormap)")
            explanation_md = gr.Markdown()

    submit_btn.click(
        fn=predict_and_explain,
        inputs=image_input,
        outputs=[label_output, gradcam_output, explanation_md],
    )

if __name__ == "__main__":
    demo.launch()

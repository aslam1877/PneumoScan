import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import os

# ----------------------------
# Load Model (cached across reruns)
# ----------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "model/pneumonia_model.pth")

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 2)
    )
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at `{MODEL_PATH}`. Please ensure the model weights are available.")
        st.stop()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# ----------------------------
# Image Transform
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

classes = ["Normal", "Pneumonia"]

# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="PneumoScan", page_icon="🫁", layout="centered")

st.title("🫁 PneumoScan")
st.subheader("AI-Powered Chest X-Ray Analysis")
st.write("Upload a chest X-ray image to check for signs of pneumonia.")

uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-Ray", use_container_width=True)

    with st.spinner("Analyzing..."):
        img = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)

        result = classes[pred.item()]
        conf = confidence.item() * 100
        normal_prob = probs[0][0].item() * 100
        pneumonia_prob = probs[0][1].item() * 100

    # --- Results ---
    st.markdown("---")

    if result == "Normal":
        st.success(f"**Prediction: {result}** ({conf:.1f}% confidence)")
    else:
        st.error(f"**Prediction: {result}** ({conf:.1f}% confidence)")

    # Probability breakdown
    st.markdown("#### Probability Breakdown")
    col1, col2 = st.columns(2)
    col1.metric("Normal", f"{normal_prob:.1f}%")
    col2.metric("Pneumonia", f"{pneumonia_prob:.1f}%")

    st.progress(normal_prob / 100, text=f"Normal: {normal_prob:.1f}%")
    st.progress(pneumonia_prob / 100, text=f"Pneumonia: {pneumonia_prob:.1f}%")

    # Low confidence warning
    if conf < 70:
        st.warning("⚠️ Low confidence prediction. This may not be a valid chest X-ray, "
                   "or the image quality is poor. Please consult a medical professional.")

    st.markdown("---")
    st.caption("**Disclaimer:** This tool is for educational purposes only. "
               "It is not a substitute for professional medical diagnosis. "
               "Always consult a qualified healthcare provider.")
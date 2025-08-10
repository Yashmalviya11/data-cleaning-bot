import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
import cv2
import torch
from diffusers import StableDiffusionPipeline

# Load YOLO model
model = YOLO("yolov8n.pt")

# Define furniture categories
OBJECT_NAMES = {56: "chair", 57: "couch", 58: "potted plant", 59: "bed", 60: "table"}

# Load AI model for interior design (Stable Diffusion)
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Detect objects in an image
def detect_objects(image_path):
    results = model(image_path)
    detected_objects = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls_id in OBJECT_NAMES:
                detected_objects.append({
                    "name": OBJECT_NAMES[cls_id],
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2)
                })
    return detected_objects

# Change wall color
def change_wall_color(image, color):
    img = np.array(image)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_wall = np.array([0, 0, 50])  # Adjust thresholds for wall selection
    upper_wall = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower_wall, upper_wall)
    img[mask > 0] = color
    return Image.fromarray(img)

# Generate AI-powered interior redesign
def generate_new_design(prompt):
    image = pipe(prompt).images[0]
    return image

# Streamlit UI
st.title("🏡 AI-Powered Virtual Interior Designer")
st.write("Upload an image of your room and explore AI-powered redesign options!")

# Upload images
room_image_file = st.file_uploader("Upload your room image", type=["jpg", "png", "jpeg"])

if room_image_file:
    room_image = Image.open(room_image_file)
    st.image(room_image, caption="Uploaded Room Image", use_column_width=True)

    # Detect furniture
    objects = detect_objects(room_image_file)
    if objects:
        st.subheader("🛋 Detected Furniture")
        for obj in objects:
            st.write(f"**{obj['name']}** - Confidence: {obj['confidence']:.2f}")

    # Color Change Option
    st.subheader("🎨 Change Wall Color")
    color = st.color_picker("Pick a new wall color", "#FFFFFF")
    if st.button("Apply Wall Color"):
        new_wall_color = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
        recolored_image = change_wall_color(room_image, new_wall_color)
        st.image(recolored_image, caption="Recolored Room", use_column_width=True)

    # AI Redesign
    st.subheader("🖌 AI-Powered Interior Redesign")
    theme = st.selectbox("Choose a Design Style", ["Modern", "Minimalist", "Boho", "Luxury"])
    if st.button("Generate New Design"):
        prompt = f"A {theme} interior design room with stylish decor and furniture"
        ai_generated_image = generate_new_design(prompt)
        st.image(ai_generated_image, caption="AI-Generated Room Design", use_column_width=True)

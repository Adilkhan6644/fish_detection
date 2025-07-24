import streamlit as st

# ----------------------------
# Streamlit App Config (MUST BE FIRST)
# ----------------------------
st.set_page_config(
    page_title="Fish Species Detector",
    layout="wide",
    page_icon="🐟"
)

from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import warnings
import os
import math

# Suppress warnings
warnings.filterwarnings("ignore")

# Create .streamlit config to avoid torch.classes errors
def create_streamlit_config():
    """Create streamlit config to suppress file watcher errors"""
    streamlit_dir = ".streamlit"
    if not os.path.exists(streamlit_dir):
        os.makedirs(streamlit_dir)
    
    config_content = """[server]
fileWatcherType = "none"
headless = true

[runner]
magicEnabled = false

[browser]
gatherUsageStats = false
"""
    
    config_path = os.path.join(streamlit_dir, "config.toml")
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            f.write(config_content)

# Create config on import
create_streamlit_config()

# Initialize model with caching to avoid reloading
@st.cache_resource
def load_model():
    """Load YOLOv8 model with caching"""
    try:
        model = YOLO("best100.pt")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def detect_measuring_board(image):
    """
    Detect measuring board/ruler in the image to establish scale
    Returns pixels per cm ratio, or None if not detected
    """
    # Convert to grayscale for line detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Use edge detection to find ruler markings
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Find horizontal lines (ruler markings)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    if lines is not None:
        # Look for parallel horizontal lines that could be ruler markings
        horizontal_lines = []
        for line in lines:
            rho, theta = line[0]
            # Check if line is roughly horizontal (theta close to 0 or pi)
            if abs(theta) < 0.2 or abs(theta - np.pi) < 0.2:
                horizontal_lines.append(rho)
        
        if len(horizontal_lines) >= 2:
            # Find the distance between markings (assuming 5cm intervals)
            horizontal_lines.sort()
            distances = [horizontal_lines[i+1] - horizontal_lines[i] for i in range(len(horizontal_lines)-1)]
            avg_distance = np.mean(distances)
            
            # If we detect regular intervals, assume they represent 5cm
            if avg_distance > 10:  # Minimum pixel distance threshold
                pixels_per_cm = avg_distance / 5.0
                return pixels_per_cm
    
    return None

def estimate_fish_size(width_px, height_px, pixels_per_cm):
    """Convert pixel dimensions to real-world size"""
    if pixels_per_cm:
        width_cm = width_px / pixels_per_cm
        height_cm = height_px / pixels_per_cm
        return width_cm, height_cm
    return None, None

def manual_scale_estimation(image_width, image_height):
    """
    Fallback method: estimate scale based on typical fish photo proportions
    This is less accurate but provides a rough estimate
    """
    # Assume a typical fish photo where the fish takes up about 1/3 of the image width
    # and estimate based on common fish sizes (15-30cm average)
    estimated_fish_width_cm = 20  # Average assumption
    assumed_fish_proportion = 0.3  # Fish takes 30% of image width
    
    pixels_per_cm = (image_width * assumed_fish_proportion) / estimated_fish_width_cm
    return pixels_per_cm

# Load model
model = load_model()

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("🐟 Fish Detection App")
st.sidebar.markdown("Upload a fish image to detect species and calculate dimensions.")

# Size estimation options
st.sidebar.markdown("---")
st.sidebar.subheader("📏 Size Estimation")
size_method = st.sidebar.selectbox(
    "Choose size estimation method:",
    ["Auto-detect measuring board", "Manual scale input", "Rough estimation"]
)

manual_scale = None
if size_method == "Manual scale input":
    manual_scale = st.sidebar.number_input(
        "Enter known object size (cm):",
        min_value=1.0, max_value=200.0, value=20.0, step=0.5,
        help="Enter the real-world size of a reference object in the image"
    )

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

# ----------------------------
# Main UI
# ----------------------------
st.title("🎣 Fish Species Detector & Measurement")
st.markdown("Detect fish species and estimate real-world size using a custom YOLOv8 model.")

# Show model info
if model:
    with st.expander("🔍 Model Information"):
        st.write(f"**Model classes:** {list(model.names.values())}")
        st.write(f"**Number of classes:** {len(model.names)}")
        st.write("**Current settings:** Confidence = 0.1 (10%), IoU = 0.4")

if model is None:
    st.error("Model failed to load. Please check your model file path.")
    st.stop()

if uploaded_file:
    try:
        # Convert and show uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        img_height, img_width = img_np.shape[:2]

        st.markdown("### 📷 Original Uploaded Image")
        st.image(image, use_container_width=True)

        # Determine scale for size estimation
        pixels_per_cm = None
        scale_method_used = "None"
        
        if size_method == "Auto-detect measuring board":
            pixels_per_cm = detect_measuring_board(img_np)
            if pixels_per_cm:
                scale_method_used = f"Auto-detected ({pixels_per_cm:.2f} pixels/cm)"
            else:
                st.warning("⚠️ Could not detect measuring board. Using rough estimation.")
                pixels_per_cm = manual_scale_estimation(img_width, img_height)
                scale_method_used = f"Rough estimation ({pixels_per_cm:.2f} pixels/cm)"
        
        elif size_method == "Manual scale input" and manual_scale:
            # For manual input, we'll need the user to click on a reference object
            # For now, we'll use a default assumption
            pixels_per_cm = img_width * 0.1 / manual_scale  # Rough calculation
            scale_method_used = f"Manual scale ({pixels_per_cm:.2f} pixels/cm)"
            st.info("💡 Manual scale estimation is approximate. For better accuracy, include a measuring board in your photo.")
        
        else:
            pixels_per_cm = manual_scale_estimation(img_width, img_height)
            scale_method_used = f"Rough estimation ({pixels_per_cm:.2f} pixels/cm)"

        st.markdown("### 🧠 Detection Results")
        st.info(f"**Scale method:** {scale_method_used}")
        
        with st.spinner("Detecting fish..."):
            # Run prediction with error handling (very low confidence threshold)
            results = model.predict(img_np, verbose=False, conf=0.1, iou=0.4)
            result = results[0]
            boxes = result.boxes
            class_names = result.names

            # Draw on image
            img_drawn = img_np.copy()

            if boxes is None or len(boxes) == 0:
                st.warning("No fish detected.")
            else:
                cols = st.columns(2)
                detection_count = 0
                
                with cols[0]:
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = class_names[cls_id]

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        width_px = x2 - x1
                        height_px = y2 - y1

                        # Calculate real-world size
                        width_cm, height_cm = estimate_fish_size(width_px, height_px, pixels_per_cm)

                        # Draw bounding box with size info
                        cv2.rectangle(img_drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Add labels with size information
                        label_text = f"{label} ({conf:.2f})"
                        if width_cm and height_cm:
                            size_text = f"L:{width_cm:.1f}cm H:{height_cm:.1f}cm"
                            cv2.putText(img_drawn, label_text, (x1, y1 - 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            cv2.putText(img_drawn, size_text, (x1, y1 - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        else:
                            cv2.putText(img_drawn, label_text, (x1, y1 - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                        # Display detailed info
                        detection_count += 1
                        st.success(f"**Detection {detection_count}: {label}**")
                        st.write(f"- Confidence: `{conf:.2f}`")
                        
                        # Pixel measurements
                        st.write(f"- **Pixel dimensions:**")
                        st.write(f"  - Width: `{width_px}` px")
                        st.write(f"  - Height: `{height_px}` px")
                        
                        # Real-world measurements
                        if width_cm and height_cm:
                            st.write(f"- **Estimated real size:**")
                            st.write(f"  - Length: `{width_cm:.1f}` cm")
                            st.write(f"  - Height: `{height_cm:.1f}` cm")
                            
                            # Calculate fish length (assuming length is the longer dimension)
                            fish_length = max(width_cm, height_cm)
                            st.write(f"  - **Total length: `{fish_length:.1f}` cm**")
                            
                            # Add weight estimation based on length (rough formula for fish)
                            # Weight (g) ≈ 0.1 × Length³ (very rough approximation)
                            estimated_weight = 0.1 * (fish_length ** 3)
                            if estimated_weight < 1000:
                                st.write(f"  - Estimated weight: `{estimated_weight:.0f}` g")
                            else:
                                st.write(f"  - Estimated weight: `{estimated_weight/1000:.1f}` kg")
                        
                        st.write(f"- Position: ({x1}, {y1}) to ({x2}, {y2})")
                        st.markdown("---")

                with cols[1]:
                    st.image(img_drawn, caption="🖼️ Detection Output with Size", use_container_width=True)

        # Size estimation disclaimer
        with st.expander("ℹ️ Size Estimation Accuracy"):
            st.markdown("""
            **Size estimation accuracy depends on the method used:**
            
            - **Auto-detect measuring board**: Most accurate when a measuring board/ruler is clearly visible
            - **Manual scale input**: Good accuracy if you provide a known reference object size
            - **Rough estimation**: Least accurate, based on typical fish proportions
            
            **Tips for better accuracy:**
            - Include a measuring board or ruler in your photos
            - Ensure the fish and measuring device are on the same plane
            - Take photos from directly above for best results
            - Use good lighting to help detect measuring board markings
            """)

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")

else:
    st.info("Please upload an image from the sidebar to begin.")
    
    # Show example of measuring board usage
    st.markdown("### 📏 For Best Size Estimation Results")
    st.markdown("""
    Include a measuring board or ruler in your fish photos. The app can automatically detect 
    measuring boards with regular markings and provide accurate size estimates.
    
    **Example measuring board features:**
    - Clear markings every 5cm
    - High contrast markings
    - Positioned parallel to the fish
    """)

# ----------------------------
# Footer
# ----------------------------
st.markdown("""
<hr style="border:1px solid #ccc">
<div style='text-align: center'>
    Built by Adil using YOLOv8 🚀 | Enhanced with size estimation 📏
</div>
""", unsafe_allow_html=True)
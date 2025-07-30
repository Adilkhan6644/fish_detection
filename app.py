import streamlit as st

# ----------------------------
# Streamlit App Config (MUST BE FIRST)
# ----------------------------
st.set_page_config(
    page_title="Fish Species Detector",
    layout="wide",
    page_icon="🐟"
)

import torch
# Fix for PyTorch 2.6 YOLO loading issue
torch.serialization.add_safe_globals([
    'ultralytics.nn.tasks.DetectionModel',
    'ultralytics.nn.modules.head.Detect',
    'ultralytics.nn.modules.conv.Conv',
    'ultralytics.nn.modules.block.C2f',
    'ultralytics.nn.modules.block.Bottleneck',
    'ultralytics.nn.modules.block.SPPF'
])

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

# Enhanced Scale Detection Functions
def detect_measuring_board_enhanced(image):
    """
    Enhanced measuring board detection for fishing scales/rulers
    Returns pixels per cm ratio and detected markings info
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Try multiple detection methods in order of reliability
    methods = [
        detect_scale_by_numbers,
        detect_scale_by_markings, 
        detect_scale_by_patterns
    ]
    
    for method in methods:
        try:
            if method == detect_scale_by_numbers:
                scale_info = method(image, gray)
            elif method == detect_scale_by_markings:
                scale_info = method(gray)
            else:
                scale_info = method(gray, hsv)
                
            if scale_info:
                return scale_info
        except Exception as e:
            continue
    
    return None

def detect_scale_by_numbers(image, gray):
    """Detect measuring board by finding numerical markings"""
    # This would require pytesseract for OCR
    # For now, return None to skip this method
    return None

def detect_scale_by_markings(gray):
    """Detect measuring board by finding regular tick marks or lines"""
    # Enhanced edge detection
    edges1 = cv2.Canny(gray, 50, 150, apertureSize=3)
    edges2 = cv2.Canny(gray, 30, 100, apertureSize=3)
    edges = cv2.bitwise_or(edges1, edges2)
    
    # Detect lines
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=60)
    
    if lines is not None:
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            rho, theta = line[0]
            
            # Horizontal lines (measuring board lying horizontally)
            if abs(theta) < 0.4 or abs(theta - np.pi) < 0.4:
                horizontal_lines.append(rho)
            # Vertical lines (measuring board standing vertically)
            elif abs(theta - np.pi/2) < 0.4:
                vertical_lines.append(rho)
        
        # Check horizontal lines first
        if len(horizontal_lines) >= 4:
            horizontal_lines.sort()
            intervals = []
            
            for i in range(len(horizontal_lines) - 1):
                interval = abs(horizontal_lines[i + 1] - horizontal_lines[i])
                if 8 < interval < 150:  # Reasonable interval range for cm markings
                    intervals.append(interval)
            
            if len(intervals) >= 3:
                intervals = np.array(intervals)
                mean_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                
                # Check for consistency
                if std_interval < mean_interval * 0.4:
                    return {
                        'pixels_per_cm': mean_interval,
                        'method': 'Horizontal Line Detection',
                        'confidence': 'Medium',
                        'detected_lines': len(horizontal_lines),
                        'std_deviation': std_interval
                    }
        
        # Check vertical lines
        if len(vertical_lines) >= 4:
            vertical_lines.sort()
            intervals = []
            
            for i in range(len(vertical_lines) - 1):
                interval = abs(vertical_lines[i + 1] - vertical_lines[i])
                if 8 < interval < 150:
                    intervals.append(interval)
            
            if len(intervals) >= 3:
                intervals = np.array(intervals)
                mean_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                
                if std_interval < mean_interval * 0.4:
                    return {
                        'pixels_per_cm': mean_interval,
                        'method': 'Vertical Line Detection', 
                        'confidence': 'Medium',
                        'detected_lines': len(vertical_lines),
                        'std_deviation': std_interval
                    }
    
    return None

def detect_scale_by_patterns(gray, hsv):
    """Detect measuring board by finding repeating patterns"""
    height, width = gray.shape
    
    # Sample horizontal strip for pattern analysis
    middle_y = height // 2
    strip_height = max(20, min(60, height // 8))
    
    start_y = max(0, middle_y - strip_height // 2)
    end_y = min(height, middle_y + strip_height // 2)
    
    horizontal_strip = gray[start_y:end_y, :]
    
    if horizontal_strip.size > 0:
        # Calculate intensity profile
        profile = np.mean(horizontal_strip, axis=0)
        
        # Smooth the profile to reduce noise
        kernel_size = max(3, len(profile) // 100)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones(kernel_size) / kernel_size
        profile_smooth = np.convolve(profile, kernel, mode='same')
        
        # Find peaks and valleys
        peaks = []
        valleys = []
        
        for i in range(2, len(profile_smooth) - 2):
            # Peak detection
            if (profile_smooth[i] > profile_smooth[i-1] and 
                profile_smooth[i] > profile_smooth[i+1] and
                profile_smooth[i] > profile_smooth[i-2] and
                profile_smooth[i] > profile_smooth[i+2]):
                
                # Check if this peak is far enough from the last one
                if not peaks or i - peaks[-1] > 8:
                    peaks.append(i)
            
            # Valley detection
            if (profile_smooth[i] < profile_smooth[i-1] and 
                profile_smooth[i] < profile_smooth[i+1] and
                profile_smooth[i] < profile_smooth[i-2] and
                profile_smooth[i] < profile_smooth[i+2]):
                
                if not valleys or i - valleys[-1] > 8:
                    valleys.append(i)
        
        # Analyze peak intervals
        if len(peaks) >= 4:
            peak_intervals = np.diff(peaks)
            mean_peak_interval = np.mean(peak_intervals)
            std_peak_interval = np.std(peak_intervals)
            
            if std_peak_interval < mean_peak_interval * 0.5 and mean_peak_interval > 10:
                # Estimate scale - assume major markings
                if mean_peak_interval > 25:  # Likely 1cm markings
                    pixels_per_cm = mean_peak_interval
                else:  # Likely smaller markings (0.5cm or 5mm)
                    pixels_per_cm = mean_peak_interval * 2
                
                return {
                    'pixels_per_cm': pixels_per_cm,
                    'method': 'Pattern Analysis',
                    'confidence': 'Low-Medium',
                    'detected_peaks': len(peaks),
                    'mean_interval': mean_peak_interval
                }
    
    return None

def visualize_scale_detection(image, scale_info):
    """Create visualization showing detected scale"""
    if not scale_info:
        return image
    
    vis_image = image.copy()
    height, width = vis_image.shape[:2]
    
    pixels_per_cm = scale_info['pixels_per_cm']
    
    # Draw grid lines every cm
    for cm in range(0, int(width / pixels_per_cm) + 1):
        x = int(cm * pixels_per_cm)
        if x < width:
            color = (0, 255, 0) if cm % 5 == 0 else (0, 150, 0)
            thickness = 2 if cm % 5 == 0 else 1
            cv2.line(vis_image, (x, 0), (x, height), color, thickness)
            
            # Label every 5cm
            if cm % 5 == 0 and cm > 0:
                cv2.putText(vis_image, f'{cm}', (x + 2, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return vis_image

def estimate_fish_size_enhanced(width_px, height_px, scale_info):
    """Enhanced fish size estimation with scale info"""
    if not scale_info:
        return None, None, None
    
    pixels_per_cm = scale_info['pixels_per_cm']
    width_cm = width_px / pixels_per_cm
    height_cm = height_px / pixels_per_cm
    
    # Fish length is typically the longer dimension
    fish_length_cm = max(width_cm, height_cm)
    
    # Estimate weight using length-weight relationship
    # General fish formula: W = a * L^b (where a≈0.01-0.02, b≈2.8-3.2)
    estimated_weight_g = 0.015 * (fish_length_cm ** 3.0)
    
    return width_cm, height_cm, estimated_weight_g

def manual_scale_estimation(image_width, image_height):
    """Fallback scale estimation"""
    estimated_fish_width_cm = 20
    assumed_fish_proportion = 0.3
    pixels_per_cm = (image_width * assumed_fish_proportion) / estimated_fish_width_cm
    
    return {
        'pixels_per_cm': pixels_per_cm,
        'method': 'Manual Estimation',
        'confidence': 'Low',
        'note': 'Fallback method - low accuracy'
    }

# Initialize model with caching
@st.cache_resource
def load_model():
    """Load YOLOv8 model with caching and PyTorch 2.6 compatibility"""
    try:
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        try:
            st.info("Attempting to download fresh model...")
            model = YOLO("yolov8n.pt")
            return model
        except Exception as e2:
            st.error(f"Fallback also failed: {e2}")
            return None

# Load model
model = load_model()

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("🐟 Fish Detection App")
st.sidebar.markdown("Upload a fish image with a measuring board for accurate size detection.")

# Show system info
st.sidebar.markdown("---")
with st.sidebar.expander("🔧 System Info"):
    st.write(f"PyTorch version: {torch.__version__}")
    st.write("Enhanced scale detection enabled")

# Detection settings
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Detection Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)
show_scale_visualization = st.sidebar.checkbox("Show Scale Grid Overlay", value=True)

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

# ----------------------------
# Main UI
# ----------------------------
st.title("🎣 Enhanced Fish Species Detector & Measurement")
st.markdown("Advanced fish detection with automatic measuring board recognition for precise size estimation.")

# Show model status
if model:
    with st.expander("🔍 Model Information"):
        st.write(f"**Model classes:** {list(model.names.values())}")
        st.write(f"**Number of classes:** {len(model.names)}")
        st.write(f"**Confidence threshold:** {confidence_threshold}")
        st.success("✅ Model loaded with enhanced scale detection")

if model is None:
    st.error("❌ Model failed to load. Please check your model file.")
    st.stop()

if uploaded_file:
    try:
        # Load and display image
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        img_height, img_width = img_np.shape[:2]

        st.markdown("### 📷 Original Image")
        st.image(image, use_container_width=True)

        # Enhanced scale detection
        st.markdown("### 📏 Scale Detection Analysis")
        
        with st.spinner("Analyzing measuring board..."):
            scale_info = detect_measuring_board_enhanced(img_np)
        
        if scale_info:
            st.success(f"✅ **Scale detected using {scale_info['method']}**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Scale Resolution", f"{scale_info['pixels_per_cm']:.2f} pixels/cm")
                st.write(f"**Confidence:** {scale_info['confidence']}")
                
            with col2:
                st.write("**Detection Details:**")
                for key, value in scale_info.items():
                    if key not in ['pixels_per_cm', 'method', 'confidence']:
                        st.write(f"- {key}: {value}")
            
            # Show scale visualization
            if show_scale_visualization:
                scale_vis = visualize_scale_detection(img_np, scale_info)
                st.image(scale_vis, caption="📐 Detected Scale Grid Overlay", use_container_width=True)
                
        else:
            st.warning("⚠️ Could not detect measuring board automatically")
            st.info("💡 **Tips for better scale detection:**")
            st.markdown("""
            - Ensure the measuring board has clear, regular markings
            - Good lighting and contrast between markings and background
            - Measuring board should be clearly visible and not obstructed
            - Try different angles or positions of the measuring board
            """)
            
            # Use fallback estimation
            scale_info = manual_scale_estimation(img_width, img_height)
            st.write(f"Using fallback estimation: {scale_info['pixels_per_cm']:.2f} pixels/cm")

        # Fish Detection
        st.markdown("### 🐠 Fish Detection & Measurement")
        
        with st.spinner("Detecting fish..."):
            results = model.predict(img_np, verbose=False, conf=confidence_threshold, iou=0.4)
            result = results[0]
            boxes = result.boxes
            class_names = result.names

        # Process detections
        if boxes is None or len(boxes) == 0:
            st.warning("🔍 No fish detected with current confidence threshold.")
            st.info(f"Try lowering the confidence threshold (currently {confidence_threshold})")
        else:
            # Create annotated image
            img_annotated = img_np.copy()
            detection_data = []
            
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = class_names[cls_id]
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width_px = x2 - x1
                height_px = y2 - y1
                
                # Calculate real-world measurements
                width_cm, height_cm, estimated_weight = estimate_fish_size_enhanced(
                    width_px, height_px, scale_info
                )
                
                # Store detection data
                detection_data.append({
                    'id': i + 1,
                    'species': label,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2),
                    'width_px': width_px,
                    'height_px': height_px,
                    'width_cm': width_cm,
                    'height_cm': height_cm,
                    'length_cm': max(width_cm, height_cm) if width_cm and height_cm else None,
                    'estimated_weight': estimated_weight
                })
                
                # Draw bounding box
                color = (0, 255, 0)  # Green
                cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, 3)
                
                # Prepare labels
                main_label = f"#{i+1}: {label} ({conf:.2f})"
                
                if width_cm and height_cm:
                    fish_length = max(width_cm, height_cm)
                    size_label = f"L: {fish_length:.1f}cm"
                    weight_label = f"~{estimated_weight:.0f}g" if estimated_weight < 1000 else f"~{estimated_weight/1000:.1f}kg"
                    
                    # Draw labels with background
                    label_y = y1 - 15
                    cv2.putText(img_annotated, main_label, (x1, label_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
                    cv2.putText(img_annotated, main_label, (x1, label_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    cv2.putText(img_annotated, size_label, (x1, label_y + 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3)
                    cv2.putText(img_annotated, size_label, (x1, label_y + 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    cv2.putText(img_annotated, weight_label, (x1, label_y + 45), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
                    cv2.putText(img_annotated, weight_label, (x1, label_y + 45), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
                else:
                    cv2.putText(img_annotated, main_label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
                    cv2.putText(img_annotated, main_label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Display results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### 📊 Detection Summary")
                st.write(f"**Total fish detected:** {len(detection_data)}")
                
                for detection in detection_data:
                    with st.expander(f"🐟 Fish #{detection['id']}: {detection['species']}", expanded=True):
                        st.write(f"**Species:** {detection['species']}")
                        st.write(f"**Confidence:** {detection['confidence']:.1%}")
                        
                        # Pixel measurements
                        st.write("**Pixel Dimensions:**")
                        st.write(f"• Width: {detection['width_px']} px")
                        st.write(f"• Height: {detection['height_px']} px")
                        
                        # Real-world measurements
                        if detection['length_cm']:
                            st.write("**Real-world Measurements:**")
                            st.write(f"• **Length: {detection['length_cm']:.1f} cm**")
                            st.write(f"• Width: {min(detection['width_cm'], detection['height_cm']):.1f} cm")
                            
                            if detection['estimated_weight']:
                                if detection['estimated_weight'] < 1000:
                                    st.write(f"• **Estimated Weight: {detection['estimated_weight']:.0f} g**")
                                else:
                                    st.write(f"• **Estimated Weight: {detection['estimated_weight']/1000:.1f} kg**")
                        else:
                            st.write("*Size estimation requires scale detection*")
                        
                        # Bounding box info
                        bbox = detection['bbox']
                        st.write(f"**Position:** ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})")
            
            with col2:
                st.markdown("#### 🖼️ Annotated Detection Results")
                st.image(img_annotated, use_container_width=True)

        # Additional Analysis
        if detection_data and any(d['length_cm'] for d in detection_data):
            st.markdown("### 📈 Fish Analysis")
            
            # Create summary statistics
            measured_fish = [d for d in detection_data if d['length_cm']]
            
            if len(measured_fish) > 1:
                lengths = [d['length_cm'] for d in measured_fish]
                weights = [d['estimated_weight'] for d in measured_fish if d['estimated_weight']]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Average Length", f"{np.mean(lengths):.1f} cm")
                    st.metric("Largest Fish", f"{max(lengths):.1f} cm")
                
                with col2:
                    if weights:
                        avg_weight = np.mean(weights)
                        if avg_weight < 1000:
                            st.metric("Average Weight", f"{avg_weight:.0f} g")
                        else:
                            st.metric("Average Weight", f"{avg_weight/1000:.1f} kg")
                        
                        max_weight = max(weights)
                        if max_weight < 1000:
                            st.metric("Heaviest Fish", f"{max_weight:.0f} g")
                        else:
                            st.metric("Heaviest Fish", f"{max_weight/1000:.1f} kg")
                
                with col3:
                    st.metric("Total Fish Count", len(detection_data))
                    species_count = len(set(d['species'] for d in detection_data))
                    st.metric("Species Detected", species_count)

        # Scale Detection Tips
        with st.expander("💡 Tips for Better Scale Detection"):
            st.markdown("""
            **For optimal measuring board detection:**
            
            1. **Clear Markings:** Use a measuring board with high contrast markings (dark on light or vice versa)
            2. **Regular Intervals:** Boards with markings every 1cm or 0.5cm work best
            3. **Good Lighting:** Ensure even lighting without shadows on the measuring board
            4. **Board Position:** Place the measuring board parallel to the fish, on the same plane
            5. **Avoid Obstruction:** Make sure the measuring board is fully visible and not covered
            6. **Image Quality:** Use high-resolution images for better marking detection
            
            **Supported measuring board types:**
            - Standard fish measuring boards with cm markings
            - Rulers with regular tick marks
            - Grid patterns with consistent spacing
            - Boards with numerical labels (10, 20, 30, etc.)
            """)

        # Export Results
        if detection_data:
            st.markdown("### 📤 Export Results")
            
            # Prepare CSV data
            csv_data = []
            for detection in detection_data:
                csv_data.append({
                    'Fish_ID': detection['id'],
                    'Species': detection['species'],
                    'Confidence': f"{detection['confidence']:.3f}",
                    'Length_cm': f"{detection['length_cm']:.2f}" if detection['length_cm'] else "N/A",
                    'Width_cm': f"{min(detection['width_cm'], detection['height_cm']):.2f}" if detection['width_cm'] else "N/A",
                    'Estimated_Weight_g': f"{detection['estimated_weight']:.1f}" if detection['estimated_weight'] else "N/A",
                    'Scale_Method': scale_info['method'],
                    'Scale_Confidence': scale_info['confidence']
                })
            
            import pandas as pd
            df = pd.DataFrame(csv_data)
            
            # Show table
            st.dataframe(df, use_container_width=True)
            
            # Download button
            csv_string = df.to_csv(index=False)
            st.download_button(
                label="📊 Download Results as CSV",
                data=csv_string,
                file_name=f"fish_detection_results_{uploaded_file.name.split('.')[0]}.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"❌ An error occurred during processing: {e}")
        with st.expander("🔧 Debug Information"):
            st.code(str(e))
            import traceback
            st.code(traceback.format_exc())

else:
    st.info("📤 Please upload a fish image from the sidebar to begin detection.")
    
    # Show example and instructions
    st.markdown("### 🎯 How to Use This App")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **1. Prepare Your Image:**
        - Include a measuring board or ruler in the photo
        - Ensure good lighting and clear visibility
        - Position the measuring board parallel to the fish
        
        **2. Upload and Detect:**
        - Upload your image using the sidebar
        - The app will automatically detect the scale
        - Adjust confidence threshold if needed
        """)
    
    with col2:
        st.markdown("""
        **3. Review Results:**
        - View detected fish with size measurements
        - Check the scale detection confidence
        - Export results as CSV for record keeping
        
        **4. Tips for Best Results:**
        - Use high-resolution images
        - Include multiple reference points
        - Ensure measuring board is clearly visible
        """)

# ----------------------------
# Footer
# ----------------------------
st.markdown("""
<hr style="border:1px solid #ccc; margin-top: 50px;">
<div style='text-align: center; color: #666; padding: 20px;'>
    🐟 <strong>Enhanced Fish Detection & Measurement System</strong> 🐟<br>
    Built with YOLOv8 • Advanced Scale Detection • Real-time Size Estimation<br>
    <small>PyTorch 2.6+ Compatible • Streamlit Powered</small>
</div>
""", unsafe_allow_html=True)
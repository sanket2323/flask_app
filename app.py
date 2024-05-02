from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        processed_file_path = process_image(filepath)
        return render_template('display.html', filename=processed_file_path)


def process_image(filepath):
    # Read the image
    image = cv2.imread(filepath)

    # Convert image to RGB (OpenCV loads images in BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the image from RGB to HSV (Hue, Saturation, Value) - better for color segmentation
    hsv_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # Define range for green color (typically representing vegetation)
    # These ranges might need to be adjusted for your specific images
    lower_green = np.array([25, 52, 72])
    upper_green = np.array([102, 255, 255])
    mask_green = cv2.inRange(hsv_img, lower_green, upper_green)

    # Define range for non-vegetative areas
    # Adjust the range for detecting areas without vegetation, if necessary
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask_red = cv2.inRange(hsv_img, lower_red, upper_red)

    # Create a mask for non-vegetation areas, assuming these don't fall in green or red
    mask_non_veg = cv2.bitwise_not(cv2.add(mask_green, mask_red))

    # Create an empty canvas and fill the masks
    output = np.zeros_like(image_rgb)
    output[mask_green != 0] = [0, 255, 0]  # Green for vegetation
    output[mask_red != 0] = [255, 0, 0]  # Red for bare ground

    # Convert output to BGR for saving with OpenCV
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    # Save the processed image
    processed_filepath = filepath.replace('uploads', 'static')
    cv2.imwrite(processed_filepath, output_bgr)

    return processed_filepath

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)

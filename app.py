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
    # Read image using OpenCV
    image = cv2.imread(filepath)

    # Convert to grayscale for simplicity in processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a mask where high values are assumed to be vegetation
    mask = cv2.inRange(gray, 150, 255)  # Adjust the range as per your data

    # Map the mask to a color scale
    color_mask = np.zeros_like(image)
    color_mask[mask > 0] = [0, 255, 0]  # Green for high vegetation
    color_mask[(mask <= 0) & (gray > 100)] = [0, 165, 255]  # Orange for moderate
    color_mask[gray <= 100] = [0, 0, 255]  # Red for low

    # Combine with the original image slightly for visual effect
    result = cv2.addWeighted(image, 0.5, color_mask, 0.5, 0)

    # Save processed image
    processed_filepath = filepath.replace('uploads', 'static')
    cv2.imwrite(processed_filepath, result)

    return processed_filepath


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)

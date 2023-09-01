import os
import glob
import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory, Response
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os
from datetime import datetime


app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = "uploads/all_class"
STATIC_FOLDER = "static"

# Load model
model = tf.keras.models.load_model(STATIC_FOLDER + "/model.h5")

IMAGE_SIZE = 300

def load_and_preprocess_image():
    test_fldr = 'uploads'
    test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
            test_fldr,
            target_size = (IMAGE_SIZE, IMAGE_SIZE),
            batch_size = 1,
            class_mode = None,
            shuffle = False)
    test_generator.reset()
    return test_generator


# Predict & classify image
def classify(model):
    batch_size = 1
    test_generator = load_and_preprocess_image()
    prob = model.predict_generator(test_generator, steps=len(test_generator)/batch_size)
    labels = {0: 'Non melanoma', 1: 'Melanoma'}
    label = labels[1] if prob[0][0] >= 0.5 else labels[0]
    classified_prob = prob[0][0] if prob[0][0] >= 0.5 else 1 - prob[0][0]
    return label, classified_prob


# home page
@app.route("/", methods=['GET'])
def home():
    filelist = glob.glob("uploads/all_class/*.*")
    for filePath in filelist:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file")
    return render_template("home.html")


@app.route("/classify", methods=["POST", "GET"])
def upload_file():

    if request.method == "GET":
        return render_template("home.html")

    else:
        file = request.files["image"]
        upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        print(upload_image_path)
        file.save(upload_image_path)

        label, prob = classify(model)
        prob = round((prob * 100), 2)

    return render_template(
        "classify.html", image_file_name=file.filename, label=label, prob=prob
    )


@app.route("/classify/<filename>")
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/one")
def index():
    return render_template("index.html")

# Initialize the camera object
cap = cv2.VideoCapture(0)

# Define the upload directory for captured frames
UPLOAD_DIR = 'uploads/all_class'

# Define a route for the video stream
def gen():
    while True:
        # Read a frame from the camera object
        ret, frame = cap.read()

        # If the frame was successfully read, convert it to a byte string
        if ret:
            frame = cv2.imencode('.jpg', frame)[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            break

# Define a route for the video feed
@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Define a route for capturing frames
@app.route('/capture_frame', methods=['POST'])
def capture_frame():
    # Read the frame from the camera object
    ret, frame = cap.read()

    # If the frame was successfully read, save it to disk
    if ret:
        # Create the upload directory if it doesn't exist
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        # Generate a unique filename for the captured frame
        filename = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}.jpg"
        filepath = os.path.join(UPLOAD_DIR, filename)

        # Save the frame to disk as a JPEG image
        cv2.imwrite(filepath, frame)

        # Classify the captured frame
        label, prob = classify(model)
        prob = round((prob * 100), 2)

        return render_template(
            "classify.html", image_file_name=filename, label=label, prob=prob
        )
    else:
        return 'Failed to capture frame!'
    
    


if __name__ == "__main__":
    app.run(host='0.0.0.0')
    app.run()
    

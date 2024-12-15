import os
import re
import io
import zlib
from werkzeug.utils import secure_filename
from flask import Response
from cs50 import SQL
from flask import Flask, flash, jsonify, redirect, render_template, request, session ,url_for, send_file
from flask_session import Session
from tempfile import mkdtemp
from werkzeug.exceptions import default_exceptions, HTTPException, InternalServerError
from werkzeug.security import check_password_hash, generate_password_hash
from datetime import datetime
import face_recognition
from PIL import Image
from base64 import b64encode, b64decode
import re
import cv2
import numpy as np
import pickle
import math
import faiss
from helpers import apology, login_required
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
from deepface import DeepFace
# Configure application
app = Flask(__name__)
#configure flask-socketio

# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Ensure responses aren't cached
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Configure CS50 Library to use SQLite database
db = SQL("sqlite:///data.db")

# Global variable
FACES_DIR = "static/face"
ENCODINGS_FILE = "face_encodings.pkl"
index = None
encodings_list = []
filenames_list = []

# Faiss index
def build_faiss_index(encodings_file):
    global index, encodings_list, filenames_list
    
    if os.path.exists(encodings_file):
        with open(encodings_file, "rb") as f:
            data = pickle.load(f)
        
        encodings_list = [item["encoding"] for item in data]
        filenames_list = [item["filename"] for item in data]
        
        # To numpy array
        encodings_array = np.array(encodings_list).astype("float32")
        
        # Create FAISS index
        index = faiss.IndexFlatL2(encodings_array.shape[1])  # L2 is Euclidean distance
        index.add(encodings_array)  # Add to index
        print(f"FAISS Index built with {len(encodings_list)} faces.")
    else:
        print("No existing encodings found. Starting fresh.")
        index = faiss.IndexFlatL2(128)  # Default is 128 dimension

# New file added handler
class NewFileHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()

    def on_created(self, event):
        if event.is_directory:
            return

        if event.src_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"New file detected: {event.src_path}")
            process_new_file(event.src_path)

def process_new_file(file_path):
    global index, encodings_list, filenames_list
    
    try:
        # Load image
        image = face_recognition.load_image_file(file_path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            # Add new encoding to list
            new_encoding = encodings[0].astype("float32")
            encodings_list.append(new_encoding)
            filenames_list.append(os.path.basename(file_path))

            # Add new encoding to FAISS index
            index.add(np.array([new_encoding]))
            
            # Update pkl
            with open(ENCODINGS_FILE, "wb") as f:
                pickle.dump([{"filename": filenames_list[i], "encoding": encodings_list[i]} 
                             for i in range(len(encodings_list))], f)
            
            print(f"File {file_path} added to encodings and FAISS index.")
        else:
            print(f"No face detected in {file_path}.")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")


@app.route("/")
@login_required
def home():
    return redirect("/home")

@app.route("/home")
@login_required
def index():
    return render_template("index.html")



@app.route("/login", methods=["GET", "POST"])
def login():
    """Log user in"""

    # Forget any user_id
    session.clear()

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":

        # Assign inputs to variables
        input_username = request.form.get("username")
        input_password = request.form.get("password")

        # Ensure username was submitted
        if not input_username:
            return render_template("login.html",messager = 1)



        # Ensure password was submitted
        elif not input_password:
             return render_template("login.html",messager = 2)

        # Query database for username
        username = db.execute("SELECT * FROM users WHERE username = :username",
                              username=input_username)

        # Ensure username exists and password is correct
        if len(username) != 1 or not check_password_hash(username[0]["hash"], input_password):
            return render_template("login.html",messager = 3)

        # Remember which user has logged in
        session["user_id"] = username[0]["id"]

        # Redirect user to home page
        return redirect("/")

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("login.html")


@app.route("/logout")
def logout():
    """Log user out"""

    # Forget any user_id
    session.clear()

    # Redirect user to login form
    return redirect("/")



@app.route("/register", methods=["GET", "POST"])
def register():
    """Register user"""

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":

        # Assign inputs to variables
        input_username = request.form.get("username")
        input_password = request.form.get("password")
        input_confirmation = request.form.get("confirmation")

        # Ensure username was submitted
        if not input_username:
            return render_template("register.html",messager = 1)

        # Ensure password was submitted
        elif not input_password:
            return render_template("register.html",messager = 2)

        # Ensure passwsord confirmation was submitted
        elif not input_confirmation:
            return render_template("register.html",messager = 4)

        elif not input_password == input_confirmation:
            return render_template("register.html",messager = 3)

        # Query database for username
        username = db.execute("SELECT username FROM users WHERE username = :username",
                              username=input_username)

        # Ensure username is not already taken
        if len(username) == 1:
            return render_template("register.html",messager = 5)

        # Query database to insert new user
        else:
            new_user = db.execute("INSERT INTO users (username, hash) VALUES (:username, :password)",
                                  username=input_username,
                                  password=generate_password_hash(input_password, method="pbkdf2:sha256", salt_length=8),)

            if new_user:
                # Keep newly registered user logged in
                session["user_id"] = new_user

            # Flash info for the user
            flash(f"Registered as {input_username}")

            # Redirect user to homepage
            return redirect("/")

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("register.html")




@app.route("/facereg", methods=["GET", "POST"])
def facereg():
    session.clear()
    if request.method == "POST":


        encoded_image = (request.form.get("pic")+"==").encode('utf-8')
        username = request.form.get("name")
        name = db.execute("SELECT * FROM users WHERE username = :username",
                        username=username)
              
        if len(name) != 1:
            return render_template("camera.html",message = 1)

        id_ = name[0]['id']    
        compressed_data = zlib.compress(encoded_image, 9) 
        
        uncompressed_data = zlib.decompress(compressed_data)
        
        decoded_data = b64decode(uncompressed_data)
        
        new_image_handle = open('./static/face/unknown/'+str(id_)+'.jpg', 'wb')
        
        new_image_handle.write(decoded_data)
        new_image_handle.close()
        try:
            image_of_bill = face_recognition.load_image_file(
            './static/face/'+str(id_)+'.jpg')
        except:
            return render_template("camera.html",message = 5)

        bill_face_encoding = face_recognition.face_encodings(image_of_bill)[0]

        unknown_image = face_recognition.load_image_file(
        './static/face/unknown/'+str(id_)+'.jpg')

        try:
            unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
        except:
            return render_template("camera.html",message = 2)

#      commpare faces
        results = face_recognition.compare_faces(
        [bill_face_encoding], unknown_face_encoding)

        if results[0]:
            username = db.execute("SELECT * FROM users WHERE username = :username",
                              username="swa")
            session["user_id"] = username[0]["id"]
            return redirect("/")
        else:
            return render_template("camera.html",message=3)


    else:
        return render_template("camera.html")



@app.route("/facesetup", methods=["GET", "POST"])
def facesetup():
    if request.method == "POST":
        encoded_image = (request.form.get("pic")+"==").encode('utf-8')

        id_=db.execute("SELECT id FROM users WHERE id = :user_id", user_id=session["user_id"])[0]["id"] 
        compressed_data = zlib.compress(encoded_image, 9) 
        
        uncompressed_data = zlib.decompress(compressed_data)
        decoded_data = b64decode(uncompressed_data)
        
        new_image_handle = open('./static/face/'+str(id_)+'.jpg', 'wb')
        
        new_image_handle.write(decoded_data)
        new_image_handle.close()
        image_of_bill = face_recognition.load_image_file(
        './static/face/'+str(id_)+'.jpg')    
        try:
            bill_face_encoding = face_recognition.face_encodings(image_of_bill)[0]
        except:    
            return render_template("face.html",message = 1)
        
        return redirect("/home")

    else:
        return render_template("face.html")

# Compare 2 faces
@app.route("/compare", methods=["GET", "POST"])
def compare_faces():
    if request.method == "POST":
        try:
            image1 = request.files.get("image1")
            image2 = request.files.get("image2")

            if not image1 or not image2:
                return jsonify({"error": "Both images are required."}), 400

            
            image1_np = np.frombuffer(image1.read(), np.uint8)
            image2_np = np.frombuffer(image2.read(), np.uint8)

            img1 = cv2.imdecode(image1_np, cv2.IMREAD_COLOR)
            img2 = cv2.imdecode(image2_np, cv2.IMREAD_COLOR)

            face_locations1 = face_recognition.face_locations(img1)
            face_locations2 = face_recognition.face_locations(img2)
            
            encodings1 = face_recognition.face_encodings(img1, face_locations1)
            encodings2 = face_recognition.face_encodings(img2, face_locations2)

            if len(encodings1) == 0 or len(encodings2) == 0:
                return jsonify({"error": "Could not detect faces in one or both images."}), 400

           
            distance = face_recognition.face_distance(encodings1, encodings2[0])
            threshold = 1.0
            normalized_distance = min(distance[0] / threshold, 1)
            alpha = 1.0
            similarity = math.exp(-normalized_distance * alpha) * 100

            return jsonify({"similarity": round(similarity, 2)})

        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return render_template("compare.html")
    
# Face detection
@app.route("/detect", methods=["GET","POST"])
def detect_faces():
    if request.method == "POST":
        try:
            image_file = request.files.get("image")

            if not image_file:
                return "No image uploaded.", 400

            
            image_np = np.frombuffer(image_file.read(), np.uint8)
            img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

           
            face_locations = face_recognition.face_locations(img)

            # Bounding box
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

            
            _, img_encoded = cv2.imencode('.png', img)
            img_io = io.BytesIO(img_encoded)

            
            return send_file(img_io, mimetype='image/png')

        except Exception as e:
            return str(e), 500
    else:
        return render_template("detection.html")


# Find image
@app.route("/finding", methods=["GET", "POST"])
def find_faces():
    if request.method == "POST":
        try:
            image = request.files.get("image")
            if not image:
                return jsonify({"error": "No image uploaded."}), 400

            # Save to unknown
            filename = secure_filename(image.filename)
            image_path = os.path.join("./static/face/unknown", filename)
            image.save(image_path)

            img = cv2.imread(image_path)
            if img is None:
                return jsonify({"error": "Error in decoding image."}), 400

            encodings_user = face_recognition.face_encodings(img)
            if len(encodings_user) == 0:
                return jsonify({"error": "No face found in the uploaded image."}), 400

            # Search in FAISS Index
            query_encoding = np.array([encodings_user[0]]).astype("float32")
            distances, indices = index.search(query_encoding, k=10)  # 10 nearest

            result = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx != -1 and dist < 1.0:  # threshold
                    result.append({"image_url": f"/static/face/{filenames_list[idx]}", "distance": dist})

            if not result:
                return jsonify({"message": "No similar faces found."})

            return render_template("finding.html", result=result, image_url=f"/{image_path}")

        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return render_template("finding.html")


@app.route('/liveness', methods=["GET","POST"])
def liveness_detection():
    if request.method == "POST":
        try:
            image_file = request.files.get('image')
            if not image_file:
                return jsonify({"error": "No image uploaded."}), 400

            np_image = np.frombuffer(image_file.read(), np.uint8)
            image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

            # Analysis
            try:
                face_objs = DeepFace.extract_faces(
                    img_path=image,
                    anti_spoofing=True 
                )

                if len(face_objs) == 0:
                    return jsonify({"result": "No face detected."}), 400
                            
                for face_obj in face_objs:
                    facial_area = face_obj.get('facial_area', None)
                    if facial_area:
                        x = facial_area.get('x', 0)
                        y = facial_area.get('y', 0)
                        w = facial_area.get('w', 0)
                        h = facial_area.get('h', 0)
                        is_real = face_obj.get("is_real", False)

                        label = "REAL" if is_real else "SPOOF"
                        color = (0, 255, 0) if is_real else (0, 0, 255)  # Green: Real | Red: Spoof
                        
                        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    else:
                        print("Invalid facial_area:", facial_area)

                _, img_encoded = cv2.imencode('.png', image)
                img_io = io.BytesIO(img_encoded)
                return send_file(img_io, mimetype='image/png')

            except Exception as e:
                return jsonify({"error": f"DeepFace error: {str(e)}"}), 500

        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return render_template("liveness.html")


def errorhandler(e):
    """Handle error"""
    if not isinstance(e, HTTPException):
        e = InternalServerError()
    return render_template("error.html",e = e)


# Listen for errors
for code in default_exceptions:
    app.errorhandler(code)(errorhandler)

if __name__ == '__main__':
    # Build FAISS Index khi khởi chạy
    build_faiss_index(ENCODINGS_FILE)
    # Setup location for observer    
    path = FACES_DIR
    event_handler = NewFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path=path, recursive=False)

    print(f"Monitoring folder: {path}")
    observer.start()
    app.run()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


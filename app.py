from configparser import ConfigParser
import os
import cv2
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    login_required,
    logout_user,
    current_user,
)
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import DataRequired
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy

# Initialize Flask app
app = Flask(__name__)


# Load configuration from properties file
def load_app_config():
    config = ConfigParser()
    config.read("config.properties")
    return config["App"]


configuration = load_app_config()

app.secret_key = configuration.get("SECRET_KEY")

# Configure SQLAlchemy for MySQL
app.config["SQLALCHEMY_DATABASE_URI"] = configuration.get("SQLALCHEMY_DATABASE_URI")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = configuration.getboolean(
    "SQLALCHEMY_TRACK_MODIFICATIONS"
)
db = SQLAlchemy(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)


# Define User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)


# Callback to reload the user object
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Define image upload form
class UploadForm(FlaskForm):
    image = FileField("Image", validators=[DataRequired()])
    submit = SubmitField("Submit")


# Load the pre-trained model
model_path = r"./model/model2.h5"
model = tf.keras.models.load_model(model_path, compile=False)


def analyze_image(image):
    # Preprocess the image
    preprocessed_image = preprocess_images([image])
    # Perform prediction using the loaded model
    prediction = model.predict(preprocessed_image)
    # Convert prediction to a string representation
    prediction_str = ", ".join(str(x) for x in prediction[0])
    class_label = np.argmax(prediction)
    # Return the string representation of the prediction
    return prediction_str, class_label


def preprocess_images(images):
    preprocessed_images = []
    for image in images:
        # Resize the image to match the expected input shape (128x128)
        print("Input image shape:", image.shape)
        resized_image = cv2.resize(image, (128, 128))

        # Check if the image is grayscale and convert it to RGB if necessary
        if len(resized_image.shape) == 2:  # Grayscale image
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)
        elif len(resized_image.shape) == 3:  # RGB image
            if (
                resized_image.shape[2] == 1
            ):  # Single-channel image (unlikely, but just in case)
                resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)
            elif resized_image.shape[2] == 4:  # RGBA image (remove alpha channel)
                resized_image = resized_image[:, :, :3]

        # Normalize pixel values to range [0, 1]
        normalized_image = resized_image / 255.0

        preprocessed_images.append(normalized_image)

    preprocessed_images = np.array(preprocessed_images)
    print("Preprocessed image shape:", preprocessed_images.shape)
    return preprocessed_images


# Route for homepage
@app.route("/")
def index():
    return render_template("index.html")


# Route for user login
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash("Logged in successfully.", "success")
            return redirect(url_for("upload"))
        else:
            flash("Invalid username or password", "error")
    return render_template("login.html")


# Route for user logout
@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out successfully.", "success")
    return redirect(url_for("index"))


# # Route for image upload
# @app.route('/upload', methods=['GET', 'POST'])
# @login_required
# def upload():
#     form = UploadForm()
#     if form.validate_on_submit():
#         f = form.image.data
#         filename = f.filename
#         f.save(os.path.join('uploads', filename))
#         image_path = os.path.join('uploads', filename)
#         image = cv2.imread(image_path)
#         result = analyze_image(image)
#         flash(result, 'success')
#         return render_template('output.html', result=result)
#     return render_template('upload.html', form=form)


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    form = UploadForm()
    if form.validate_on_submit():
        f = form.image.data
        filename = f.filename
        f.save(os.path.join("uploads", filename))
        image_path = os.path.join("uploads", filename)
        image = cv2.imread(image_path)
        result, class_label = analyze_image(
            image
        )  # Modify analyze_image to also return the class label
        flash(result, "success")
        classes = ["mild", "Moderate", "No_DR", "proliferate_DR", "Severe"]
        return render_template(
            "output.html", result=result, class_label=classes[class_label]
        )  # Pass the class_label to the template
    return render_template("upload.html", form=form)


# Route for user registration
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if User.query.filter_by(username=username).first():
            flash("Username already exists", "error")
        else:
            password_hash = generate_password_hash(password)
            new_user = User(username=username, password_hash=password_hash)
            db.session.add(new_user)
            db.session.commit()
            flash("Registration successful. Please log in.", "success")
            return redirect(url_for("login"))
    return render_template("register.html")


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=configuration.getboolean("DEBUG"), port=configuration.getint("PORT"))

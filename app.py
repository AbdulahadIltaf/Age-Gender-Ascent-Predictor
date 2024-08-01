from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
import os
from io import BytesIO
import librosa
import numpy as np
import pyaudio
import wave
import threading
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

app = Flask(__name__, static_folder='assets', template_folder='.')

# Ensure the uploads directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Audio recording parameters
audio_format = pyaudio.paInt16
channels = 1
rate = 44100
chunk = 1024
recording_thread = None
frames = []
is_recording = False

# Load data (assuming CSV format)
train = pd.read_csv('Ntrain.csv')
test = pd.read_csv('Ntest.csv')

train = train.drop(columns='Unnamed: 0')
test = test.drop(columns='Unnamed: 0')

# Label encode the categorical columns
label_encoders = {}
for column in ['age', 'gender', 'accent']:
    le = LabelEncoder()
    train[column] = le.fit_transform(train[column])
    test[column] = le.transform(test[column])
    label_encoders[column] = le

# Separate features and target variables
X_train = train.drop(columns=['age', 'gender', 'accent'])
y_train_age = train['age']
y_train_gender = train['gender']
y_train_accent = train['accent']

X_test = test.drop(columns=['age', 'gender', 'accent'])
y_test_age = test['age']
y_test_gender = test['gender']
y_test_accent = test['accent']

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Determine unique class labels
unique_age_labels = y_train_age.unique()
unique_gender_labels = y_train_gender.unique()
unique_accent_labels = y_train_accent.unique()

# Define class weights for imbalanced classes
age_class_weights = {label: 1 / count for label, count in zip(unique_age_labels, y_train_age.value_counts())}
gender_class_weights = {label: 1 / count for label, count in zip(unique_gender_labels, y_train_gender.value_counts())}
accent_class_weights = {label: 1 / count for label, count in zip(unique_accent_labels, y_train_accent.value_counts())}

# Initialize classifiers with class weights
age_clf = RandomForestClassifier(class_weight=age_class_weights)
gender_clf = RandomForestClassifier(class_weight=gender_class_weights)
accent_clf = RandomForestClassifier(class_weight=accent_class_weights)

# Train the classifiers
age_clf.fit(X_train_scaled, y_train_age)
gender_clf.fit(X_train_scaled, y_train_gender)
accent_clf.fit(X_train_scaled, y_train_accent)

# Initialize regression models
age_regressor = LinearRegression()
gender_regressor = LinearRegression()
accent_regressor = LinearRegression()

# Train the regressors
age_regressor.fit(X_train_scaled, y_train_age)
gender_regressor.fit(X_train_scaled, y_train_gender)
accent_regressor.fit(X_train_scaled, y_train_accent)

def predict_features(features_dict):
    """
    Predicts age, accent, and gender from the given feature dictionary using both classifiers and regressors.
    
    Args:
        features_dict (dict): A dictionary of feature values.
        
    Returns:
        list: A list containing the predicted age, accent, and gender as strings.
    """
    # Convert dictionary to list in the order of feature names from X_train
    feature_names = X_train.columns
    features = [features_dict.get(name, 0) for name in feature_names]  # default to 0 if feature not in dict
    
    # Ensure features are in the correct shape (1, -1) for prediction
    features = np.array(features).reshape(1, -1)
    
    # Scale the features
    features_scaled = scaler.transform(features)
    
    # Apply feature-based predictions first
    age_prediction = predict_age(features[0]) if 'predict_age' in globals() else None
    gender_prediction = predict_gender(features[0]) if 'predict_gender' in globals() else None
    accent_prediction = predict_accent(features[0]) if 'predict_accent' in globals() else None
    
    # Predict using classifiers
    age_pred_class = age_clf.predict(features_scaled)[0]
    gender_pred_class = gender_clf.predict(features_scaled)[0]
    accent_pred_class = accent_clf.predict(features_scaled)[0]
    
    # Predict using regressors
    age_pred_reg = age_regressor.predict(features_scaled)[0]
    gender_pred_reg = gender_regressor.predict(features_scaled)[0]
    accent_pred_reg = accent_regressor.predict(features_scaled)[0]
    
    # Convert regressor predictions to categorical values (if needed)
    age_pred_reg_class = np.round(age_pred_reg).astype(int)
    gender_pred_reg_class = np.round(gender_pred_reg).astype(int)
    accent_pred_reg_class = np.round(accent_pred_reg).astype(int)
    
    # Decode categorical predictions
    age_str_class = label_encoders['age'].inverse_transform([age_pred_class])[0]
    age_str_reg = label_encoders['age'].inverse_transform([age_pred_reg_class])[0]
    
    gender_str_class = label_encoders['gender'].inverse_transform([gender_pred_class])[0]
    gender_str_reg = label_encoders['gender'].inverse_transform([gender_pred_reg_class])[0]
    
    accent_str_class = label_encoders['accent'].inverse_transform([accent_pred_class])[0]
    accent_str_reg = label_encoders['accent'].inverse_transform([accent_pred_reg_class])[0]
    
    # Compare predictions and determine final result
    age_str = age_str_class if age_str_class == age_str_reg else "Uncertain"
    gender_str = gender_str_class if gender_str_class == gender_str_reg else "Uncertain"
    accent_str = accent_str_class if accent_str_class == accent_str_reg else "Uncertain"
    
    return [age_str, gender_str, accent_str]

def extract_acoustic_features(file_path):
    audio, sample_rate = librosa.load(file_path)
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mean_mfccs = np.mean(mfccs, axis=1)  # Calculate mean across MFCC coefficients
    
    # Extract other features
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sample_rate)
    pitch = np.mean(pitches[pitches > 0])
    intensity = np.mean(librosa.feature.rms(y=audio))
    duration = librosa.get_duration(y=audio, sr=sample_rate)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate))
    
    # Additional features
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sample_rate))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=audio))
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate))
    
    return  {
        'pitch': pitch,
        'intensity': intensity,
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': spectral_bandwidth,
        'spectral_contrast': spectral_contrast,
        'spectral_rolloff': spectral_rolloff,
        'zero_crossing_rate': zero_crossing_rate,
        'chroma_stft': chroma_stft,
        'mfcc_1': mean_mfccs[0],  # Add mean MFCC coefficients as features
        'mfcc_2': mean_mfccs[1],
        'mfcc_3': mean_mfccs[2],
        # Add more MFCC coefficients or other features as needed
    }

def save_audio(file_storage, filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file_storage.save(file_path)
    return file_path

def load_audio_from_file(file_storage):
    file_bytes = BytesIO(file_storage.read())
    audio, sr = librosa.load(file_bytes, sr=None)
    return audio, sr




app.secret_key = '9b4c5a3f9eab7c41d5e917c7b79de2f2'  # Required for flash messages

password_path = 'password.txt'  # Update this path as needed

@app.route('/send_email', methods=['POST'])
def handle_form_submission():
    name = request.form.get('name')
    email = request.form.get('email')
    comment = request.form.get('comment')

    try:
        send_email(name, email, comment)
        flash('Your message has been sent successfully!', 'success')
    except Exception as e:
        flash(f'An error occurred: {e}', 'danger')

    return redirect(url_for('index'))

def read_smtp_credentials():
    with open(password_path, 'r') as file:
        lines = file.readlines()
        smtp_username = lines[0].strip()
        smtp_password = lines[1].strip()
    return smtp_username, smtp_password

def send_email(name, email, comment):
    smtp_username, smtp_password = read_smtp_credentials()

    # Create the email message
    msg = MIMEMultipart()
    msg['From'] = smtp_username
    msg['To'] = "iltafabdulahad@gmail.com"  # Receiver's email
    msg['Subject'] = f"New Contact Form Submission from {name}"

    # Compose the body of the email
    body = f"Name: {name}\nEmail: {email}\n\nMessage:\n{comment}"
    msg.attach(MIMEText(body, 'plain'))

    try:
        # Connect to the SMTP server
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()  # Secure the connection
            server.login(smtp_username, smtp_password)

            # Send the email
            server.sendmail(smtp_username, "iltafabdulahad@gmail.com", msg.as_string())
            print("Email sent successfully!")

    except Exception as e:
        print(f"Error: {e}")



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = save_audio(file, 'upload.wav')
    features = extract_acoustic_features(file_path)
    predictions = predict_features(features)
    os.remove(file_path)
    print(predictions)

    return jsonify({
        'age': predictions[0],
        'gender': predictions[1],
        'accent': predictions[2]
    })

def record_audio():
    global is_recording, frames

    audio = pyaudio.PyAudio()
    stream = audio.open(format=audio_format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
    frames = []

    while is_recording:
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(os.path.join(UPLOAD_FOLDER, 'record.wav'), 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(audio_format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

@app.route('/start_record', methods=['GET'])
def start_record():
    global is_recording, recording_thread

    if not is_recording:
        is_recording = True
        recording_thread = threading.Thread(target=record_audio)
        recording_thread.start()
        return jsonify({'message': 'Recording started'})
    else:
        return jsonify({'error': 'Already recording'})

@app.route('/stop_record', methods=['GET'])
def stop_record():
    global is_recording

    if is_recording:
        is_recording = False
        recording_thread.join()

        features = extract_acoustic_features(os.path.join(UPLOAD_FOLDER, 'record.wav'))
        predictions = predict_features(features)
        os.remove(os.path.join(UPLOAD_FOLDER, 'record.wav'))
        print(predictions)

        return jsonify({
            'message': 'Recording stopped',
            'age': predictions[0],
            'gender': predictions[1],
            'accent': predictions[2]
        })
    else:
        return jsonify({'error': 'Not currently recording'})

if __name__ == '__main__':
    app.run(debug=True)

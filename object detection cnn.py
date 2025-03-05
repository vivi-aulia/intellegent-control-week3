import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load model yang telah dilatih
model = load_model('cnn_model.h5')

# Direktori dataset
train_dir = "dataset/seg_train/seg_train"
val_dir = "dataset/seg_test/seg_test"

# Load label kelas
class_labels = ['building', 'forest', 'glacier', 'mountain', 'sea', 'street']

cap = cv2.VideoCapture(0)

def calculate_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    
    if brightness < 30:
        return brightness, "Gelap"
    elif 30 <= brightness < 100:
        return brightness, "Redup"
    else:
        return brightness, "Terang"

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Mode Night Vision dengan konversi ke skala abu-abu dan peningkatan kecerahan
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    night_vision = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    
    # Preprocessing gambar
    img = cv2.resize(frame, (150, 150))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Prediksi kelas
    pred = model.predict(img)
    label = class_labels[np.argmax(pred)]
    
    # Hitung tingkat pencahayaan
    brightness, brightness_level = calculate_brightness(frame)
    brightness_text = f'Brightness: {brightness:.2f} ({brightness_level})'
    
    # Tampilkan hasil
    cv2.putText(frame, f'Class: {label}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, brightness_text, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow('Frame', frame)
    cv2.imshow('Night Vision', night_vision)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

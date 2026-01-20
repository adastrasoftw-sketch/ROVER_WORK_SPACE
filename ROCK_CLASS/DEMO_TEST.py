import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model(r"C:\Users\Sanjay\Downloads\loki\rock_classifier_jetson.h5")
class_names = ["gravel", "brick", "pebble"]
IMG_SIZE = 224

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # 🔥 FIXED SAFE ROI
    crop_w = int(w * 0.5)
    crop_h = int(h * 0.4)
    x1 = int((w - crop_w) / 2)
    y1 = int(h - crop_h)

    roi = frame[y1:y1+crop_h, x1:x1+crop_w]

    roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi_norm = roi_resized / 255.0
    roi_input = np.expand_dims(roi_norm, axis=0)

    preds = model.predict(roi_input, verbose=0)
    class_id = np.argmax(preds)
    confidence = preds[0][class_id]

    if confidence < 0.70:
        label = "UNCERTAIN"
        color = (0, 0, 255)
    else:
        label = f"{class_names[class_id]} ({confidence*100:.1f}%)"
        color = (0, 255, 0)

    cv2.rectangle(frame, (x1, y1), (x1+crop_w, y1+crop_h), color, 2)
    cv2.putText(frame, label, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Rock Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
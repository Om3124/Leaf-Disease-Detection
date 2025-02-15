import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model = load_model(r'C:\Users\Om\Documents\Leaf disease dect\plant_disease_model.h5')

class_labels = ['Healthy', 'Powdery', 'Rustt', ...]  

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break
  
    img = cv2.resize(frame, (150, 150))
   
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0) 
    img_array /= 255.0 

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]  
    predicted_label = class_labels[predicted_class]
    
    cv2.putText(frame, f"Predicted: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Plant Disease Detection', frame)
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

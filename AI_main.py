# Иницилизация библиотек и настроек
import cv2
from keras.models import load_model
import numpy as np
model = load_model('keras_model.h5')
labels = ['...', '...', '...']
cap = cv2.VideoCapture(0)
while True:
    # Захват изображения
    success, imageOrig = cap.read()
    if success:
        # Обработка изображения
        image = cv2.resize(imageOrig, (224, 224))
        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        name = labels[prediction.argmax(axis=1)[0]]
        cv2.putText(imageOrig, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.imshow("Result image", imageOrig)
        # Обработка выхода
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        # Проверка на ошибки
        print("Ошибка!")
        break
# Закрытие всех окон
cv2.destroyAllWindows()
cap.release()

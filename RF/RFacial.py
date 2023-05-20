import cv2

# Carga el clasificador de caras de OpenCV
face_cascade = cv2.CascadeClassifier('/RF/haarcascade_frontalface_default.xml')

# Inicia la cámara
cap = cv2.VideoCapture(0)

while True:
    # Captura una imagen de la cámara
    ret, img = cap.read()

    # Convierte la imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detecta las caras en la imagen
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Dibuja un rectángulo alrededor de cada cara detectada
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    # Muestra la imagen en una ventana
    cv2.imshow('img',img)

    # Espera a que se presione la tecla 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la cámara y cierra la ventana
cap.release()
cv2.destroyAllWindows()


import os
from tkinter import Tk, filedialog
import cv2

def select_image():
    """
    Permite al usuario seleccionar una imagen desde su sistema de archivos.
    """
    Tk().withdraw()  # Ocultar la ventana principal de Tkinter
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    return file_path

def capture_image(output_path):
    """
    Captura una imagen desde la cámara y la guarda.
    """
    cap = cv2.VideoCapture(0)
    print("Presiona 's' para guardar la imagen o 'q' para salir.")
    while True:
        ret, frame = cap.read()
        cv2.imshow("Captura de imagen", frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite(output_path, frame)
            print(f"Imagen guardada en {output_path}")
            break
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    option = input("Selecciona una opción: 1 para cargar imagen, 2 para capturar con cámara: ")
    if option == "1":
        image_path = select_image()
        print(f"Imagen seleccionada: {image_path}")
    elif option == "2":
        output_path = "captured_image.jpg"
        capture_image(output_path)

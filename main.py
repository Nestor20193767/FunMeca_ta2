import mediapipe as mp
import streamlit as st
import tempfile
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math

# Clase encargada de la detección y conteo de repeticiones
class BenchPressCounter:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.rep_count = 0
        self.direction = None  # 'up' o 'down'
        self.threshold_up = 170  # Umbral para la posición más alta (Momento 2)
        self.threshold_down = 95  # Umbral para la posición más baja (Momento 1)
        self.in_rep = False  # Variable para controlar si estamos en una repetición


    # Rescalamiento de pixeles a cm
    def rescalarPx_Cm(self, valor, pix=294, cm='52', mode='3dec'):
        cm = float(cm)
        # Calcular la escala de píxeles a centímetros
        escala = cm / pix
        # Rescalar el valor en píxeles a centímetros
        valor = valor * escala

        # Determinar cuántos decimales mostrar según el modo
        if mode == 'float':
            # Devolver como un float con decimales
            return valor
        elif mode == 'int':
            # Redondear a entero
            return round(valor)
        elif mode == '2dec':
            # Redondear a 2 decimales
            return round(valor, 2)
        elif mode == '3dec':
            # Redondear a 3 decimales
            return round(valor, 3)
        else:
            # Si el modo no es reconocido, devolver el valor original
            return valor

    # Función para calcular la distancia entre dos puntos
    def calculate_distance(self, p1, p2):
            return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    # Función para calcular el ángulo entre tres puntos (p1, p2, p3)
    def calculate_angle(self, p1, p2, p3):
        # Calcular los vectores
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])

        # Calcular el ángulo entre los vectores
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        angle = np.arccos(dot_product / (norm_v1 * norm_v2))
        return math.degrees(angle)

    # Función para procesar video y detectar movimientos
    def process_video(self, uploaded_file, ancho):
        # Crear un archivo temporal para almacenar el video
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        # Abre el video usando av
        video = av.open(tfile.name)
        stframe = st.empty()

        with self.mp_pose.Pose(min_detection_confidence=0.65, min_tracking_confidence=0.65) as pose:
            for frame in video.decode(video=0):
                # Convertir el frame a un array de numpy
                image = np.array(frame.to_image())
                height, width, _ = image.shape

                # Procesar la imagen para detectar poses
                results = pose.process(image)

                # Dibujar el esqueleto en la imagen
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    # Detectar las posiciones de los puntos relevantes (hombros, codos, muñecas)
                    shoulder_left = landmarks[11]  # Hombro izquierdo
                    elbow_left = landmarks[13]  # Codo izquierdo
                    wrist_left = landmarks[15]  # Muñeca izquierda

                    shoulder_right = landmarks[12]  # Hombro derecho
                    elbow_right = landmarks[14]  # Codo derecho
                    wrist_right = landmarks[16]  # Muñeca derecha

                    nose = landmarks[0]  # Nariz

                    # Calcular el ángulo entre el brazo y el antebrazo (izquierdo)
                    angle_left = self.calculate_angle(shoulder_left, elbow_left, wrist_left)
                    # Calcular el ángulo entre el brazo y el antebrazo (derecho)
                    angle_right = self.calculate_angle(shoulder_right, elbow_right, wrist_right)

                    # Calcular el punto de la recta entre los codos con la X de la nariz
                    x_nose = nose.x * width
                    y_nose = nose.y * height

                    # Interpolación entre los codos izquierdo y derecho para el punto sobre la recta
                    x_left = elbow_left.x * width
                    y_left = elbow_left.y * height
                    x_right = elbow_right.x * width
                    y_right = elbow_right.y * height


                    # Interpolamos un punto sobre la recta entre los codos que tenga la misma coordenada X que la nariz
                    if x_left != x_right:  # Evitar división por cero
                        # Encontramos el punto con la misma X que la nariz
                        t = (x_nose - x_left) / (x_right - x_left)
                        y_point = y_left + t * (y_right - y_left)
                        point_on_line = np.array([x_nose, y_point])
                    else:
                        point_on_line = np.array([x_left, y_left])

                     # Calcular las distancias desde el punto en la recta a cada codo
                    dist_left = self.calculate_distance(point_on_line, np.array([x_left, y_left]))
                    dist_right = self.calculate_distance(point_on_line, np.array([x_right, y_right]))

                    # Determinar si la repetición está ocurriendo (momento 1 a momento 2 o viceversa)
                    if angle_left <= self.threshold_down and angle_right <= self.threshold_down:
                        # Momento 1: Barra cerca del pecho
                        if self.direction != 'down' and not self.in_rep:
                            self.direction = 'down'
                            self.in_rep = True
                    elif angle_left >= self.threshold_up and angle_right >= self.threshold_up:
                        # Momento 2: Barra completamente extendida
                        if self.direction == 'down' and self.in_rep:
                            self.rep_count += 1
                            self.direction = 'up'
                            self.in_rep = False


                    # Interpolación entre las muñecas izquierda y derecha para el punto sobre la recta
                    x_left2 = wrist_left.x * width
                    y_left2 = wrist_left.y * height
                    x_right2 = wrist_right.x * width
                    y_right2 = wrist_right.y * height

                    # Interpolamos un punto sobre la recta entre los codos que tenga la misma coordenada X que la nariz
                    if x_left2 != x_right2:  # Evitar división por cero
                        # Encontramos el punto con la misma X que la nariz
                        t2 = (x_nose - x_left2) / (x_right2 - x_left2)
                        y_point2 = y_left2 + t2 * (y_right2 - y_left2)
                        point_on_line2 = np.array([x_nose, y_point2])
                    else:
                        point_on_line2 = np.array([x_left2, y_left2])

                    # Calcular las distancias desde el punto en la recta a cada codo
                    dist_left2 = self.calculate_distance(point_on_line2, np.array([x_left2, y_left2]))
                    dist_right2 = self.calculate_distance(point_on_line2, np.array([x_right2, y_right2]))

                    # ---------------------------------

                    # Interpolación entre los hombros izquierda y derecha para el punto sobre la recta
                    x_left3 = shoulder_left.x * width
                    y_left3 = shoulder_left.y * height
                    x_right3 = shoulder_right.x * width
                    y_right3 = shoulder_right.y * height

                    # Interpolamos un punto sobre la recta entre los codos que tenga la misma coordenada X que la nariz
                    if x_left3 != x_right3:  # Evitar división por cero
                        # Encontramos el punto con la misma X que la nariz
                        t3 = (x_nose - x_left3) / (x_right3 - x_left3)
                        y_point3 = y_left3 + t3 * (y_right3 - y_left3)
                        point_on_line3 = np.array([x_nose, y_point3])
                    else:
                        point_on_line3 = np.array([x_left3, y_left3])

                    # Calcular las distancias desde el punto en la recta a cada codo
                    dist_left3 = self.calculate_distance(point_on_line3, np.array([x_left3, y_left3]))
                    dist_right3 = self.calculate_distance(point_on_line3, np.array([x_right3, y_right3]))

                    # ---------------------------------

                    # Dibujar los ángulos en la imagen
                    img_pil = Image.fromarray(image)
                    draw = ImageDraw.Draw(img_pil)

                    # Cargar una fuente TrueType y ajustar el tamaño
                    font_path = "C://Users//ASUS//OneDrive//Escritorio//arial_bold.ttf"  # Cambia esta ruta a donde esté tu archivo de fuente
                    font = ImageFont.truetype(font_path, 40)  # Cambiar el tamaño de la fuente aquí

                    # Aumentar el tamaño del texto y colocarlo junto al codo izquierdo
                    draw.text((int(elbow_left.x * width) + 10, int(elbow_left.y * height) - 40),
                              f"{int(angle_left)}°", fill=(255, 255, 255), font=font)

                    # Aumentar el tamaño del texto y colocarlo junto al codo derecho
                    draw.text((int(elbow_right.x * width) + 10, int(elbow_right.y * height) - 40),
                              f"{int(angle_right)}°", fill=(255, 255, 255), font=font)

                    # Mostrar el conteo de repeticiones en la esquina superior izquierda
                    draw.text((10, 10), f"Repeticiones con \n"+"Alta Hipertrofia \n" +
                              f": {self.rep_count}", fill=(255, 255, 255), font=font)

                    # Dibujar puntos del esqueleto
                    self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                    # Dibujar circunferencias alrededor de los puntos
                    # Convertir las coordenadas a píxeles
                    x_shoulder_left, y_shoulder_left = int(shoulder_left.x * width), int(shoulder_left.y * height)
                    x_elbow_left, y_elbow_left = int(elbow_left.x * width), int(elbow_left.y * height)
                    x_wrist_left, y_wrist_left = int(wrist_left.x * width), int(wrist_left.y * height)

                    x_shoulder_right, y_shoulder_right = int(shoulder_right.x * width), int(shoulder_right.y * height)
                    x_elbow_right, y_elbow_right = int(elbow_right.x * width), int(elbow_right.y * height)
                    x_wrist_right, y_wrist_right = int(wrist_right.x * width), int(wrist_right.y * height)

                    # Radio de la circunferencia
                    radius = 10

                    # Dibujar las circunferencias para el brazo izquierdo
                    draw.ellipse([(x_shoulder_left - radius, y_shoulder_left - radius),
                                  (x_shoulder_left + radius, y_shoulder_left + radius)], outline="yellow", width=2)
                    draw.ellipse([(x_elbow_left - radius, y_elbow_left - radius),
                                  (x_elbow_left + radius, y_elbow_left + radius)], outline="yellow", width=2)
                    draw.ellipse([(x_wrist_left - radius, y_wrist_left - radius),
                                  (x_wrist_left + radius, y_wrist_left + radius)], outline="yellow", width=2)

                    # Dibujar las circunferencias para el brazo derecho
                    draw.ellipse([(x_shoulder_right - radius, y_shoulder_right - radius),
                                  (x_shoulder_right + radius, y_shoulder_right + radius)], outline="yellow", width=2)
                    draw.ellipse([(x_elbow_right - radius, y_elbow_right - radius),
                                  (x_elbow_right + radius, y_elbow_right + radius)], outline="yellow", width=2)
                    draw.ellipse([(x_wrist_right - radius, y_wrist_right - radius),
                                  (x_wrist_right + radius, y_wrist_right + radius)], outline="yellow", width=2)

                    # Codos ---------------------------------------------------------------------------------

                    # Calcular el punto medio entre la nariz y el codo izquierdo
                    mid_point_left = np.array([(x_nose + x_left) / 2, (y_nose + y_left) / 2])

                    # Calcular el punto medio entre la nariz y el codo derecho
                    mid_point_right = np.array([(x_nose + x_right) / 2, (y_nose + y_right) / 2])

                    # Creamos un font distinto
                    font2 = ImageFont.truetype(font_path, 20)  # Cambiar el tamaño de la fuente aquí

                    # Escribir las distancias en los puntos medios de las rectas
                    draw.text((int(mid_point_left[0]) - 20, int(mid_point_left[1]) - 40),
                              f"{self.rescalarPx_Cm(int(dist_left),cm=ancho)} cm", fill=(255, 120, 0), font=font2)

                    draw.text((int(mid_point_right[0]) - 20, int(mid_point_right[1]) - 40),
                              f"{self.rescalarPx_Cm(int(dist_right),cm=ancho)} cm", fill=(255, 120, 0), font=font2)

                    # Dibujar la recta entre los codos
                    draw.line([int(x_left), int(y_left), int(x_right), int(y_right)], fill="red", width=4)

                    # Dibujar el punto en la recta
                    draw.ellipse([(point_on_line[0] - 5, point_on_line[1] - 5),
                                  (point_on_line[0] + 5, point_on_line[1] + 5)], fill="green")

                    #  Muñecas ------------------------------------------------------------

                    # Calcular el punto medio entre la nariz y la muñeca izquierdo
                    mid_point_left2 = np.array([(x_nose + x_left2) / 2, (y_nose + y_left2) / 2])

                    # Calcular el punto medio entre la nariz y la muñeca derecha
                    mid_point_right2 = np.array([(x_nose + x_right2) / 2, (y_nose + y_right2) / 2])

                    # Escribir las distancias en los puntos medios de las rectas de la muñecas
                    draw.text((int(mid_point_left2[0]) - 20, int(mid_point_left2[1]) - 60),
                              f"{self.rescalarPx_Cm(int(dist_left2),cm=ancho)} cm", fill=(0, 120, 255), font=font2)

                    draw.text((int(mid_point_right2[0]) - 20, int(mid_point_right2[1]) - 60),
                              f"{self.rescalarPx_Cm(int(dist_right2),cm=ancho)} cm", fill=(0, 120, 255), font=font2)

                    # Dibujar la recta entre las muñecas
                    draw.line([int(x_left2), int(y_left2), int(x_right2), int(y_right2)], fill="blue", width=4)

                    # Dibujar el punto en la recta de las muñecas
                    draw.ellipse([(point_on_line2[0] - 5, point_on_line2[1] - 5),
                                  (point_on_line2[0] + 5, point_on_line2[1] + 5)], fill="green")

                    # Mostrar la imagen con esqueleto, ángulos y circunferencias
                    stframe.image(img_pil, use_column_width=True)

                    # Hombros ------------------------------------------------------------

                    # Calcular el punto medio entre la nariz y la muñeca izquierdo
                    mid_point_left3 = np.array([(x_nose + x_left3) / 2, (y_nose + y_left3) / 2])

                    # Calcular el punto medio entre la nariz y la muñeca derecha
                    mid_point_right3 = np.array([(x_nose + x_right3) / 2, (y_nose + y_right3) / 2])

                    # Escribir las distancias en los puntos medios de las rectas de la muñecas
                    draw.text((int(mid_point_left3[0]) - 20, int(mid_point_left3[1]) + 40),
                              f"{self.rescalarPx_Cm(int(dist_left3), cm=ancho)} cm", fill=(255, 255, 0), font=font2)

                    draw.text((int(mid_point_right3[0]) - 20, int(mid_point_right3[1]) + 40),
                              f"{self.rescalarPx_Cm(int(dist_right3), cm=ancho)} cm", fill=(255, 255, 0), font=font2)

                    # Dibujar la recta entre las muñecas
                    draw.line([int(x_left3), int(y_left3), int(x_right3), int(y_right3)], fill="yellow", width=4)

                    # Dibujar el punto en la recta de las muñecas
                    draw.ellipse([(point_on_line3[0] - 5, point_on_line3[1] - 5),
                                  (point_on_line3[0] + 5, point_on_line3[1] + 5)], fill="green")

                    # Verificacion de hipertrofia ----------------------------------------------------

                        # Vereficacion en Muñecas
                    percent_muñecas = (100*dist_right2)/dist_right3 # %
                    if percent_muñecas < 100:
                        modo_m = "extensores del codo"
                    elif percent_muñecas > 100:
                        modo_m = "hombro y pectoral mayor"
                    elif percent_muñecas == 100:
                        modo_m = "equilibrado"

                        # Verificación en Codos
                    percent_codo = (100 * dist_right) / dist_right3  # %
                    if percent_codo < 100:
                        modo_c = "extensores del codo"
                    elif percent_codo > 100:
                        modo_c = "hombro y pectoral mayor"
                    elif percent_codo == 100:
                        modo_c = "equilibrado"

                    # Mostrar texto en la esquina inferior derecha
                    text_derecha = "Hipertrofia generada por codos: "+ modo_c + "\n" +"Hipertrofia generada por Agarre: "+ modo_m
                    offset_x_derecha = 650  # Espacio desde el borde derecho
                    offset_y_derecha = 60  # Espacio desde el borde inferior
                    draw.text((width - offset_x_derecha ,
                               height - offset_y_derecha),
                              text_derecha, fill=(255, 255, 255), font=font2)

                       # Verificacion final

                    if modo_c == modo_m:
                        hipertrofia_final = modo_m
                    elif modo_m != modo_c:
                        hipertrofia_final = "equilibrado"

                    # Mostrar texto en la esquina inferior izquierda
                    text_izquierda = "Hipertrofia Final: \n" + hipertrofia_final
                    offset_x_izquierda = 10  # Espacio desde el borde izquierdo
                    offset_y_izquierda = 60  # Espacio desde el borde inferior
                    draw.text((offset_x_izquierda, height - offset_y_izquierda),
                              text_izquierda, fill=(255, 255, 255), font=font2)

                    # Mostrar la imagen con esqueleto, ángulos y circunferencias
                    stframe.image(img_pil, use_column_width=True)


# Clase para la interfaz de usuario
class AppFrontend:
    def __init__(self, counter):
        self.counter = counter
        self.st = st

    def run(self):
        self.st.title("Evaluación Biomecanica del Bench Press")
        uploaded_file = self.st.file_uploader("Sube un video", type=['mp4', 'mov', 'avi'])
        ancho_hombros = self.st.text_input("Escribe el cancho de los hombros en cm", value='52')

        if uploaded_file is not None:
            self.st.text("Procesando video...")
            self.counter.process_video(uploaded_file, ancho_hombros)
            self.st.text(f"Repeticiones contadas: {self.counter.rep_count}")


if __name__ == '__main__':
    # Crear una instancia de la clase BenchPressCounter
    counter = BenchPressCounter()
    app = AppFrontend(counter)
    app.run()

import cv2
import numpy as np

# Глобальные переменные для хранения координат вершин
vertices = []

# Функция обработки событий клика мыши
def mouse_click(event, x, y, flags, param):
    global vertices
    # Если клик левой кнопкой мыши, сохраняем координаты
    if event == cv2.EVENT_LBUTTONDOWN:
        vertices.append((x, y))
        # Рисуем точку на кадре
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Video', frame)
        # Если получили все 4 точки, рисуем четырехугольник и выводим координаты
        if len(vertices) == 4:
            draw_and_display(frame)

# Функция для рисования четырехугольника и вывода координат
def draw_and_display(frame):
    global vertices
    # Рисуем четырехугольник
    cv2.polylines(frame, [np.array(vertices)], isClosed=True, color=(255, 255, 255), thickness=2)
    # Выводим координаты на экран
    for i, (x, y) in enumerate(vertices):
        cv2.putText(frame, f'({x}, {y})', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow('Video', frame)

# Открываем видеофайл
video_capture = cv2.VideoCapture('VID_20240330_164210.mp4')

# Ожидаем, пока видео не откроется
while not video_capture.isOpened():
    video_capture = cv2.VideoCapture('VID_20240330_164210.mp4')
    cv2.waitKey(1000)
    print("Wait for the video to open")

# Считываем первый кадр
ret, frame = video_capture.read()

# Создаем окно с видео и устанавливаем обработчик событий мыши
cv2.imshow('Video', frame)
cv2.setMouseCallback('Video', mouse_click)

# Обработка кадров видео
while True:
    # Считываем следующий кадр
    ret, frame = video_capture.read()
    if not ret:
        break

    # Ожидаем нажатия клавиши 'q' для выхода
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Освобождаем ресурсы
video_capture.release()
cv2.destroyAllWindows()

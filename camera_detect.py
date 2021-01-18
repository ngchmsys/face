from pathlib import Path
import cv2

# 検知器のパス
CASCADE_BASE_DIR = Path("./Lib/site-packages/cv2/data/")
cascade_files = [file for file in CASCADE_BASE_DIR.glob('*.xml')]
print(f'{"-"*50}\n 検知器一覧\n{"-"*50}')
[print(f'{i} : {file.name}') for i,file in enumerate(cascade_files)]
cascade_path =  cascade_files[7]
print('-' * 50)
print(f'検知器Path： {cascade_path}')

cascade = cv2.CascadeClassifier(str(cascade_path))
color = (0, 255, 0)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rect = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))
    if len(rect) > 0:
        for x, y, w, h in rect:
            cv2.rectangle(frame, (x, y), (x+w, y+h), color)
    cv2.imshow('detected', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

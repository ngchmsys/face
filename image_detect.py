from pathlib import Path
import cv2

# 検知器のパス
CASCADE_BASE_DIR = Path("./Lib/site-packages/cv2/data/")
cascade_files = [file for file in CASCADE_BASE_DIR.glob('*.xml')]
print(f'{"-"*50}\n 検知器一覧\n{"-"*50}')
[print(f'{i} : {file.name}') for i,file in enumerate(cascade_files)]
cascade_path =  cascade_files[5]
print('-' * 50)
print(f'検知器Path： {cascade_path}')

# 顔画像のパス
img_path = "./images/ai_face_001.jpg"
src = cv2.imread(img_path, 1) # 0: gray, 1: color(default), -1: original

#gray = cv2.cvtColor(src, cv2.cv2.COLOR_BAYER_BG2GRAY)
cascade = cv2.CascadeClassifier(str(cascade_path))
#rect = cascade.detectMultiScale(gray)
rect = cascade.detectMultiScale(src)
print(f'検出範囲：{rect}')

bgr = (0, 255, 0) # (B, G, R)
if len(rect) > 0:
    for x, y, w, h in rect:
        cv2.rectangle(src, (x,y), (x+w,y+h), bgr)

cv2.imshow('FaceDetect', src)
cv2.waitKey(0)
cv2.destroyAllWindows()

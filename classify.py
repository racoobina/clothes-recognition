# импорт библиотек
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
# задаем аргументы
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to label binarizer")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# загружаем обученную модель, бинаризатор меток и изображение
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())
image = cv2.imread(args["image"])
output = image.copy()

# форматируем изображение для обработки
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

#классифицируем
prediction = model.predict(image)[0]
idx = np.argmax(prediction)
label = lb.classes_[idx]

# вывода результата: correct или incorrect
filename = args["image"][args["image"].rfind(os.path.sep) + 1:]
correct = "correct" if filename.rfind(label) != -1 else "incorrect"
print("{}: {:.2f}%, {}".format(label, prediction[idx] * 100, correct))

#Задаем в терминале следующие парамеры:
# python classify.py --model clothes_classification.model --labelbin lb.pickle --image examples/dress.jpg
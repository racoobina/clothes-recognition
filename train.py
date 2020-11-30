# Импорт необходимых библиотек
import matplotlib
matplotlib.use("Agg")
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from model.svggnet import sVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# задаем аргументы
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True)
ap.add_argument("-m", "--model", required=True)
ap.add_argument("-l", "--labelbin", required=True)
ap.add_argument("-p", "--plot", type=str, default="plot.png")
args = vars(ap.parse_args())
data = []
labels = []

# задаем количество эпох, скорость обучения, размер пакетов, и размеры изображения
Epochs = 100
LR = 1e-3
BatchSize = 32
Image_size = (96, 96, 3)

# перемешиваем пути к изображениям
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)
#print(imagePaths[1])

# проходимся по каждому изображению
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	# устранение ошибок связанных с некоректными данными
	try:
		image = cv2.resize(image, (Image_size[1], Image_size[0]))
	except cv2.error as e:
		print('поврежденное изображение!')
		print(imagePath)
	image = img_to_array(image)
	data.append(image)
 
	# записываем метки изображений исходя из его пути
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

# масштабируем интенсивность пикселей в отрезок [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# бинаризируем метки
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# разделение выборки на обучающую и тестовую
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.2, random_state=42)

# инициализация модели
model = sVGGNet.build(width=Image_size[1], height=Image_size[0],
	depth=Image_size[2], classes=len(lb.classes_))
opt = Adam(lr=LR, decay=LR / Epochs)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# расширение масива данных генератором
generator = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")
# обучение модели
H = model.fit_generator(
	generator.flow(trainX, trainY, batch_size=BatchSize),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BatchSize,
	epochs=Epochs, verbose=1)

# сохранение модели  и бинаризатора меток на диск
model.save(args["model"])
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(lb))
f.close()

# график точности и потерь
plt.style.use("ggplot")
plt.figure()
N = Epochs
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])

# Задаем в терминале следующие параметры:
# python train.py --dataset dataset --model clothes_classification.model --labelbin lb.pickle
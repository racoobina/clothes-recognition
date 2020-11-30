

# Импорт необходимых пакетов
from requests import exceptions
import argparse
import requests
import cv2
import os

# Парсер
ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required=True,
                help="Запрос для поиска в Bing")
ap.add_argument("-o", "--output", required=True,
                help="Директория для сохранения результатов")
args = vars(ap.parse_args())

# Microsoft Cognitive Services API key и количество требуемых изображений
API_KEY = "432a37b0111d4e0f8cae067f533918e7"
MAX_RESULTS = 250
GROUP_SIZE = 50

# установка API URL
URL = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"

# Для пропуска ошибок
EXCEPTIONS = set([IOError,
                  exceptions.RequestException, exceptions.HTTPError,
                  exceptions.ConnectionError, exceptions.Timeout])

# Хранение запросов в удобной переменной и установка заголовков
term = args["query"]
headers = {"Ocp-Apim-Subscription-Key": API_KEY}
params = {"q": term, "offset": 0, "count": GROUP_SIZE}

# создание запроса
print("поиск Bing API для '{}'".format(term))
search = requests.get(URL, headers=headers, params=params)
search.raise_for_status()

# Количество полученных результатов
results = search.json()
estNumResults = min(results["totalEstimatedMatches"], MAX_RESULTS)
print("Итого {} результатов для '{}'".format(estNumResults, term))

# Счетчик сохраненных фото
total = 0

# Проходимся по каждому из количества полученных результатов
for offset in range(0, estNumResults, GROUP_SIZE):
    # Обновление параметров и сохранение
    print("Запрос для группы {}-{} из {}...".format(
        offset, offset + GROUP_SIZE, estNumResults))
    params["offset"] = offset
    search = requests.get(URL, headers=headers, params=params)
    search.raise_for_status()
    results = search.json()
    print("Сохранение результатов для группы {}-{} из {}...".format(
        offset, offset + GROUP_SIZE, estNumResults))

    # Проходимся по каждому результату
    for v in results["value"]:
        # Пытаемся скачать
        try:
            # Запрос на скачивание
            print("запрос get: {}".format(v["contentUrl"]))
            r = requests.get(v["contentUrl"], timeout=30)

            # Путь для изображения
            ext = v["contentUrl"][v["contentUrl"].rfind("."):]
            p = os.path.sep.join([args["output"], "{}{}".format(
                str(total).zfill(8), ext)])

            # запись на диск
            f = open(p, "wb")
            f.write(r.content)
            f.close()

        # Пропускаем ошибки
        except Exception as e:
            if type(e) in EXCEPTIONS:
                print("пропускаем: {}".format(v["contentUrl"]))
                continue

        # обновление счетчика
        total += 1
# Использованные команды
# python search_bing_api.py --query "jeans" --output dataset/jeans
# python search_bing_api.py --query "jackets and suits" --output dataset/jackets
# python search_bing_api.py --query "underwear" --output dataset/underwear
# python search_bing_api.py --query "t-shirt" --output dataset/t-shirt
# python search_bing_api.py --query "shorts" --output dataset/shorts
# python search_bing_api.py --query "dress" --output dataset/dress
# python search_bing_api.py --query "outerwear" --output dataset/outerwear
# python search_bing_api.py --query "Jumpers, sweaters and cardigans" --output dataset/Jumpers
# python search_bing_api.py --query "overalls" --output dataset/overalls
# python search_bing_api.py --query "рубашка" --output dataset/shirts
# python search_bing_api.py --query "socks, stockings and tights" --output dataset/socks
# python search_bing_api.py --query "sweatshirts" --output dataset/sweatshirts
# python search_bing_api.py --query "swimwear and beachwear" --output dataset/swimwear
# python search_bing_api.py --query "top" --output dataset/top
# python search_bing_api.py --query "tunic" --output dataset/tunic

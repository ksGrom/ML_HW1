Домашнее задание 1.

Задача: линейная регрессия для предсказания стоимости автомобилей, реализация веб-сервиса с обученной моделью на FastAPI.

**Что было сделано.**
1. Прежде всего, данные, представленные в исходных таблицах (тренировочном и тестовом датасетах), были приведены в подходящий для обработки вид. Так, например, в колонке *mileage* значения необходимо было перевести в одну единицу измерения (kmpl). Некоторые сложности возникли с парсингом столбца *torque*, поскольку строки в нем не имели единого формата: пришлось подбирать регулярные выражения, которые позволяют извлечь численные значения признаков *torque* и *max_torque_rpm* и единицы измерения из большей части представленных строк. В результате из оригинального столбца *torque* удалось спарсить почти всё (за исключением 5-10 строк).

2. Строки с одинаковым признаковым описанием были удалены. Пропуски в столбцах были заменены на медианы. 

3. Были построены точечные диаграммы для каждой из пар численных столбцов (*seaborn.pairplot*) для выявления корреляций признаков с целевой переменной и между признаками. Подробное описание есть в .ipynb-ноутбуке.

4. Была обучена модель на вещественных признаках по аналитической формуле, с помощью GridSearchCV подобран лучший параметр для L1-регуляризации. Обученная по аналитической формуле модель показала немного лучший результат на тестовом датасете (коэф. детерминации 0.601 против 0.575). L1-регуляризация занулила 4 веса: у признаков mileage, engine, torque и seats. С помощью GridSearchCV были также найдены лучшие параметры для *ElasticNet*: оптимальное значение одного из них *l1_ratio=1.0*, что соответствует L1-регуляризации.

5. Была обучена модель на вещественных и категориальных признаках (категориальные кодируются с помощью one-hot encoding). Качество предсказаний немного улучшилось (r2=0.620).

6. Значение кастомной метрики (доля предиктов, отличающихся от реальных цен на авто не более чем на 10%), скорее всего, не удовлетворило бы заказчика: всего лишь 0.235.

7. Лучшая модель и обученные на тренировочном датасете imputer (заполняет пропуски медианами), normalizer (стандартизирует признаки), one_hot_encoder (кодирует категориальные признаки) были сохранены в .pickle-файл для дальнейшего использования модели в сервисе.

8. Был разработан сервис на FastAPI. Доступны три эндпоинта (каждый получает данные методом *post*): *"/predict_item"*, *"/predict_items"*, *"/predict_csv"*. Первый получает на вход один объект (признаковое описание) и возвращает для него предсказанную цену (float). Второй получает на вход список объектов и возвращает для них список предсказанных цен. Третий получает на вход csv-файл с объектами и возвращает такой же файл с добавленным в него столбцом с предсказанными ценами.

Скриншоты работы сервиса:
![1](./screenshots/1.jpg)
![2](./screenshots/2.jpg)
![3](./screenshots/3.jpg)
![4](./screenshots/4.jpg)
![5](./screenshots/5.jpg)
![6](./screenshots/6.jpg)
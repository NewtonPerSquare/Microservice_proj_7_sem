# Soft Sensor Microservices

Распределённая микросервисная система для поддержки дипломной работы по построению виртуального датчика качества на основе данных технологического процесса.

Система реализована на Python (FastAPI + Dash), развёртывается в Docker через `docker compose` и включает:

- `collector` — сервис сбора данных (батчи из набора данных);
- `storage` — хранилище сырых/обработанных данных (in-memory + CSV);
- `ml_service` — сервис обучения и инференса моделей (простая модель + soft sensor на RandomForest + SDVAE soft sensor);
- `web_master` — API-gateway и сервис сценариев использования;
- `visualization` — интерактивный дашборд на Plotly Dash.

---

## Требования

- Установлен **Docker** и **Docker Compose**.
- Клонирован репозиторий проекта

- В каталоге data/ лежат исходные данные для soft sensor:
  - DataSet1_60T.csv — основной CSV-файл с временным рядом;
  - DataSet1.json — JSON с метаинформацией по колонкам (ColumnKind: CV/MV/DV)

## Как начать
В корневом каталоге выполнить

    docker compose build
    docker compose up

После запуска будут доступны:

Web Master (API-gateway): http://localhost:8000 (/docs для Swagger)

Collector: http://localhost:8001 (/docs здесь и далее аналогично)

Storage: http://localhost:8002

ML Service: http://localhost:8003

Visualization (Dash-дэшборд): http://localhost:8004

Остановить систему можно либо с помощью 
docker compose down, либо послав сигнал ctrl+c


## Основные сервисы
1. Collector

GET /health — состояние сервиса и наличие датасета.

POST /batch — выдаёт батч данных из заранее загруженного набора

Collector ведёт лог запросов в файл /app/logs/collector.log 

2. Storage

GET /health — состояние хранилища (количество элементов, наличие файла)

POST /store — сохранить новые объекты

данные накапливаются в памяти и периодически сохраняются в /data/storage.csv

GET /items (если реализован) — получить текущие элементы хранилища

Логи пишутся в /app/logs/storage.log

3. ML Service

Основные группы эндпоинтов:

- Базовая демо-модель:

POST /train_from_file — обучение простой модели на демо-датасете;

POST /predict — предсказание по входному x (используется в demo-сценарии)

- RandomForest soft sensor:

POST /softsensor/train

Обучение soft sensor

POST /softsensor/predict_last - предсказания на последних n_last точках датасета

SDVAE soft sensor:

POST /softsensor_sdvae/train

POST /softsensor_sdvae/predict_last

Логи ML Service — /app/logs/ml_service.log

4. Web Master

Web Master агрегирует сценарии использования:

GET /health — состояние системы

POST /scenario/demo  - Demo-сценарий:

Запросить батч у Collector - сохранить в Storage - отправить в MLService для предсказания - вернуть предсказания и статистику (collected, stored_total, used_trained_model)

POST /scenario/train_model - запуск обучения простой модели в MLService

POST /scenario/softsensor_train - запустить обучение RandomForest soft sensor

POST /scenario/softsensor_eval - получить y_true/y_pred soft sensor для последних n_last точек

POST /scenario/softsensor_sdvae_train, POST /scenario/softsensor_sdvae_eval - аналогичные сценарии для SDVAE soft sensor

GET /report/softsensor - генерация HTML-отчёта по soft sensor (RF) на последних n_last точках: считает MSE и R², возвращает готовую HTML-страницу

5. Visualization (Dash)

Дашборд.

Кнопки:

"Запустить demo (x,y)" — вызывает /scenario/demo;

"Обучить простую модель (x,y)" — /scenario/train_model;

"Обучить soft sensor (RandomForest)" — /scenario/softsensor_train;

"Обучить soft sensor (SDVAE)" — /scenario/softsensor_sdvae_train;

"Оценить RF soft sensor и построить график" — /scenario/softsensor_eval;

"Оценить SDVAE soft sensor и построить график" — /scenario/softsensor_sdvae_eval

Поле "число последних точек" — управляет параметром n_last для оценочных сценариев

Блок "Логи" — текстовая история сценариев с timestamp’ами

График y_true vs y_pred — отображает качество soft sensor на последних n_last точках

Ссылка "открыть HTML-отчёт RF soft sensor" — открывает GET /report/softsensor?n_last=... в новой вкладке

## Соответствие требованиям задания

### 1. Архитектура и модульность (до 20 баллов)

#### 1(a). Обязательная часть — независимые сервисы, Dockerfile, docker-compose

**Статус:** ✅ Выполнено

- Реализовано 5 независимых сервисов: `collector`, `storage`, `ml_service`, `web_master`, `visualization`.
- У каждого сервиса:
  - свой `Dockerfile`;
  - свой `config.yaml` с параметрами среды;
  - отдельная директория с кодом
- Есть общий `docker-compose.yml`, который поднимает все сервисы в единой сети

#### 1(b). Дополнительная часть

- **Масштабируемость: несколько источников или разных моделей**

  **Статус:** ✅ Выполнено  

  В `ml_service` реализовано несколько моделей:

  - простая модель для demo-сценария;
  - **RandomForest soft sensor**;
  - **SDVAE/MUDVAE soft sensor**.

  Архитектура позволяет добавлять новые модели отдельными эндпоинтами без изменения других сервисов.

- **Открытость: GitHub Actions или иное CI/CD**

  **Статус:** ⛔ Не выполнено

---

### 2. Collector (до 10 баллов)

#### 2(a). Обязательная часть — API батчей и обработка ошибок

**Статус:** ✅ Выполнено

- `GET /health` — состояние сервиса и наличие датасета.
- `POST /batch` — возвращает последовательный батч данных из заранее загруженного набора.
- При отсутствии/ошибке датасета корректно фиксируются ошибки в логах.

#### 2(b). Дополнительная часть

- **Сбор не из файла, а из внешнего источника**

  **Статус:** ⛔ Не выполнено  


- **Расширенный лог действий (запись в файл)**

  **Статус:** ✅ Выполнено  

  Логи записываются в `/app/logs/collector.log` (можно посмотреть через `docker exec`)

---

### 3. Storage (до 10 баллов)

#### 3(a). Обязательная часть — API CRUD операций

**Статус:** ✅ Выполнено (в рамках упрощённой модели хранения)

- `GET /health` — состояние хранилища.
- `POST /store` — запись новых объектов (сырые данные от Collector).
- `GET /items` (если включён) — чтение текущих объектов.
- Данные хранятся in-memory и периодически сохраняются в `/data/storage.csv`.

#### 3(b). Дополнительная часть

- **Специализированные базы данных**

  **Статус:** ⛔ Не выполнено

- **Расширенный лог действий (запись в файл)**

  **Статус:** ✅ Выполнено  

  Логи Storage пишутся в `/app/logs/storage.log`

---

### 4. ML Service (до 20 баллов)

#### 4(a). Обязательная часть — API обучения, инференса и summary моделей

**Статус:** ✅ Выполнено

Реализованы основные группы эндпоинтов:

- Базовая демо-модель:
  - `POST /train_from_file`
  - `POST /predict`
- **RandomForest soft sensor**:
  - `POST /softsensor/train`
  - `POST /softsensor/predict_last`
- **SDVAE soft sensor**:
  - `POST /softsensor_sdvae/train`
  - `POST /softsensor_sdvae/predict_last`

В ответах train-эндпоинтов возвращаются:

- `rows_used`, `n_features`, `mse`, `r2`, `model_path` — это и есть summary-состояние моделей

#### 4(b). Дополнительная часть

- **Сложность модели**

  **Статус:** ✅ Выполнено  

  Помимо простой модели, реализованы:

  - `RandomForestRegressor` как soft sensor;
  - сложная модель SDVAE + MUDVAE (PyTorch, вариационный автоэнкодер).

- **Версионность моделей**

  **Статус:** ⛔ Не выполнено  

- **Хранилище моделей на основе БД**

  **Статус:** ⛔ Не выполнено

---

### 5. Web Master (до 10 баллов)

#### 5(a). Обязательная часть — API сценариев использования

**Статус:** ✅ Выполнено

Реализованы сценарии:

- `POST /scenario/demo` — demo-конвейер Collector → Storage → MLService;
- `POST /scenario/train_model` — обучение простой модели;
- `POST /scenario/softsensor_train` — обучение RF soft sensor;
- `POST /scenario/softsensor_eval` — оценка RF soft sensor (возвращает `y_true`, `y_pred`);
- `POST /scenario/softsensor_sdvae_train` — обучение SDVAE soft sensor;
- `POST /scenario/softsensor_sdvae_eval` — оценка SDVAE soft sensor;
- `GET /report/softsensor` — HTML-отчёт по RF soft sensor (MSE, R² на последних `n_last` точках)

Web Master выполняет роль API-gateway для всех сценариев

#### 5(b). Дополнительная часть

- **Аутентификация/авторизация**

  **Статус:** ⛔ Не выполнено  

- **Разделение сценариев по группам пользователей**

  **Статус:** ⛔ Не выполнено  

---

### 6. Visualization (до 20 баллов)

#### 6(a). Обязательная часть — интерактивный GUI

**Статус:** ✅ Выполнено

Сервис `visualization` реализован на Plotly Dash:

- Кнопки для вызова сценариев Web Master:
  - demo, обучение простой модели;
  - обучение/оценка RF soft sensor;
  - обучение/оценка SDVAE soft sensor.
- Поле ввода `Число последних точек (n_last)`
- Интерактивный график `y_true vs y_pred`
- Лог действий с временными метками

GUI полностью взаимодействует с Web Master по HTTP и служит прослойкой между пользователем и API.

#### 6(b). Дополнительная часть

- **Сложность GUI: дашборд, отчёты (HTML)**

  **Статус:** ✅ Выполнено  

  - Реализован полноценный дашборд с несколькими зонами (кнопки, лог, график).
  - Есть HTML-отчёт:
    - Web Master генерирует HTML через `/report/softsensor`;
    - в дашборде есть ссылка «Открыть HTML-отчёт RF soft sensor», учитывающая выбранный `n_last`.

- **Поддержка и визуализация истории выполнения сценариев**

  **Статус:** ✅ Частично выполнено  

  - В дашборде есть блок «Логи», где каждое действие добавляется с timestamp и описанием.
  - История хранится в рамках текущей сессии клиента (в Dash), отдельной персистентной БД истории нет.

---

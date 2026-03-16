# Music Recommendation System

Гибридная рекомендательная система для музыкального сервиса.

Система сочетает несколько подходов:

* **Collaborative Filtering (ALS)**
* **Learning-to-Rank (LightGBM)**
* **Online пользовательские сигналы**
* **Fallback рекомендации популярных треков**

Рекомендации выдаются через **FastAPI-микросервис**.

---

# Architecture

Рекомендательная система использует гибридную архитектуру.

```
User Request (user_id)
        │
        ▼
Recommendation Service (FastAPI)
        │
        ├── Offline recommendations (ALS)
        │
        ├── Online signals (user history)
        │
        └── Popular fallback
        │
        ▼
Hybrid ranking
        │
        ▼
Top-K recommendations
```

---

# System Design

```
                    ┌──────────────────────────┐
                    │   Raw interaction data   │
                    │  tracks / events / items │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   Data preprocessing     │
                    │ cleaning / filtering     │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Candidate generation     │
                    │ ALS / similar / popular  │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Feature engineering      │
                    │ candidate features       │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Ranking model            │
                    │ LightGBM Ranker          │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Saved artifacts          │
                    │ models / parquet recs    │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ FastAPI service          │
                    │ online recommendation    │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Top-K recommendations    │
                    └──────────────────────────┘
```

Система разделена на два контура:

**Offline pipeline**

* подготовка данных
* генерация кандидатов
* обучение моделей
* генерация рекомендаций
* сохранение артефактов

**Online сервис**

* загрузка подготовленных артефактов
* обработка пользовательского запроса
* выдача рекомендаций

---

# Project Structure

```
recsys/

 ├── data/
 │    ├── events.parquet
 │    └── items.parquet
 │
 ├── features/
 │
 ├── models/
 │
 └── recommendations/


recommendations.ipynb
recommendations_service.py
test_service.py

run_service.sh
run_tests.sh

requirements.txt
README.md
```

---

# ML Pipeline

## Data preparation

Используются данные:

```
events.parquet
items.parquet
```

Содержат:

* историю взаимодействий пользователей
* информацию о треках

---

## Candidate generation

Генерируются кандидаты рекомендаций:

* ALS collaborative filtering
* similar items
* popular tracks

---

## Feature engineering

Формируются признаки кандидатов для модели ранжирования.

---

## Ranking model

Используется **LightGBM Ranker**, который ранжирует кандидатов рекомендаций.

---

## Offline recommendations

ALS используется для генерации персональных рекомендаций пользователей.

---

# Results

В ходе экспериментов были оценены три типа рекомендаций:

* **Top Popular** — рекомендации на основе популярности треков
* **Personal ALS** — персонализированные рекомендации, полученные с помощью модели ALS
* **Final** — итоговые рекомендации после ранжирования кандидатов

Оценка проводилась по метрикам:

* Precision@10
* Recall@10
* Coverage@10
* Novelty@10

## Model comparison

| Model        | Precision@10 | Recall@10  | Coverage@10 | Novelty@10 |
| ------------ | ------------ | ---------- | ----------- | ---------- |
| Top Popular  | 0.0027       | 0.0027     | 0.00001     | 11.20      |
| Personal ALS | 0.0135       | 0.0191     | 0.00455     | 12.47      |
| Final        | **0.0211**   | **0.0261** | **0.00667** | **12.57**  |

---

## Top Popular

Модель Top Popular использует глобальную популярность треков и рекомендует одинаковый список всем пользователям.

* precision@10 ≈ 0.0027
* recall@10 ≈ 0.0027
* coverage@10 ≈ 0.00001
* novelty@10 ≈ 11.20

Модель показывает самое низкое качество рекомендаций и используется как **базовый бейзлайн**.

---

## Personal ALS

Модель **ALS (Alternating Least Squares)** реализует collaborative filtering.

* precision@10 ≈ 0.0135
* recall@10 ≈ 0.0191
* coverage@10 ≈ 0.00455
* novelty@10 ≈ 12.47

По сравнению с бейзлайном:

* precision вырос примерно в **5 раз**
* recall вырос примерно в **7 раз**
* существенно увеличилось покрытие каталога

ALS позволяет учитывать **скрытые предпочтения пользователей**.

---

## Final model

Итоговая модель использует **двухэтапную архитектуру рекомендаций**:

1. генерация кандидатов
2. ранжирование кандидатов

Результаты:

* precision@10 ≈ 0.0211
* recall@10 ≈ 0.0261
* coverage@10 ≈ 0.00667
* novelty@10 ≈ 12.57

По сравнению с ALS:

* precision увеличился примерно на **55%**
* recall увеличился примерно на **35%**
* coverage каталога также увеличился

Использование ранжирующей модели позволяет **лучше упорядочивать кандидатов и повышать релевантность рекомендаций**.

---

# Setup

## Clone repository

```
git clone https://github.com/yandex-praktikum/mle-project-sprint-4-v001.git
cd mle-project-sprint-4-v001
```

---

# Virtual Environment

Создание окружения:

```
python3 -m venv env_recsys_start
```

Активация:

```
source env_recsys_start/bin/activate
```

---

# Install dependencies

```
pip install -r requirements.txt
```

---

# Generate models and recommendations

Модели и файлы рекомендаций **не хранятся в репозитории**, так как имеют большой размер.

Они генерируются при выполнении ноутбука:

```
recommendations.ipynb
```

---

# Running the recommendation service

Запуск сервиса:

```
./run_service.sh
```

или

```
python -m uvicorn recommendations_service:app --host 0.0.0.0 --port 8000
```

Сервис будет доступен по адресу:

```
http://127.0.0.1:8000
```

---

# API

## Health check

```
GET /health
```

Ответ:

```
{"status":"ok"}
```

---

## Get recommendations

GET:

```
/recommendations?user_id=3&k=10
```

POST:

```
curl -X POST http://127.0.0.1:8000/recommendations \
-H "Content-Type: application/json" \
-d '{"user_id":3,"k":10}'
```

---

# Testing

Тестирование выполняется скриптом:

```
test_service.py
```

Проверяются три сценария:

1 пользователь без персональных рекомендаций
2 пользователь с персональными рекомендациями
3 пользователь с онлайн историей

---

# Run tests

```
./run_tests.sh
```

или

```
python test_service.py
```

Сохранить лог:

```
python test_service.py > test_service.log
```

---

# API Documentation

FastAPI автоматически генерирует документацию.

Откройте:

```
http://127.0.0.1:8000/docs
```

---

# License

Проект выполнен в рамках курса
**Онлайн-магистратура «Data Science в экономике»**

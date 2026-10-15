# 🏎️ Formula 1 Tyre Degradation & Real-Time Strategy System
### AI and Visualization Lab Project (JKU Linz - Fall 2026)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastF1](https://img.shields.io/badge/telemetry-FastF1-red.svg)](https://github.com/theOehrly/Fast-F1)
[![Course](https://img.shields.io/badge/course-AI%20%26%20Vis-brightgreen.svg)](https://jku-vds-lab.at/)

This repository contains the end-to-end data pipeline, physics-informed machine learning models, and real-time contingency strategy engine for analyzing **tyre degradation and pit stop dynamics in Formula 1**, focusing on **Red Bull Racing (Max Verstappen vs Sergio Perez)** across the 2023–2024 seasons.

---

## 📂 Структура репозитория

```
AIandVis/
├── data/
│   └── processed/
│       └── f1_curated_telemetry.csv  # 9,202 реальных круга FastF1 по 7 Гран-при (единый датасет!)
├── src/
│   ├── data/
│   │   └── f1_data.py               # Загрузчик FastF1 + генератор физически точных сценариев
│   ├── models/
│   │   ├── tyre_model.py            # Модель деградации + детектор обрыва шин (Tyre Cliff)
│   │   └── trained_tyre_model.joblib # Обученный артефакт модели (R² = 0.724, MAE = 0.68s)
│   ├── strategy/
│   │   ├── strategy_engine.py       # Движок реалтайм-стратегии с ветвлением («What-If»)
│   │   └── pit_dynamics.py          # Моделирование потерь на пит-лейн и прогрева холодных шин
│   └── simulation/
│       └── race_simulator.py        # Пошаговый симулятор реплея телеметрии гонки
├── notebooks/
│   ├── phase1_visual_analytics.py   # Диагностические графики невязок и таймлайна (Фаза I)
│   └── red_bull_analysis.py         # Сравнительный анализ стилей пилотирования (VER vs PER)
├── ai_logs/
│   └── phase1_ai_log_template.md    # Обязательный шаблон лога промптов для AI-aided пары
├── visualizations/                  # Сгенерированные графики и аналитика
├── scripts/
│   ├── download_real_data.py        # Пакетная выгрузка данных FastF1
│   └── train_model.py               # Обучение и валидация модели
├── test_pipeline.py                 # Сквозной скрипт проверки работоспособности
├── requirements.txt                 # Зафиксированные зависимости Python
├── PHASE1_GUIDE.md                  # Подробное руководство по проведению Фазы I
└── README.md
```

---

## 🚀 Быстрый старт (Инструкция для команды)

### 1. Клонирование репозитория
```bash
git clone <URL_ВАШЕГО_РЕПОЗИТОРИЯ>
cd AIandVis
```

### 2. Настройка виртуального окружения
```bash
# Создание окружения
python3 -m venv .venv

# Активация:
# на macOS / Linux:
source .venv/bin/activate
# на Windows:
# .venv\Scripts\activate

# Установка зависимостей
pip install -r requirements.txt
```

> **Обратите внимание:** Датасет (`data/processed/f1_curated_telemetry.csv`) и обученная модель (`src/models/trained_tyre_model.joblib`) **уже включены в репозиторий**. Никому из членов команды не нужно ничего скачивать часами или заново тренировать — всё готово к анализу из коробки!

---

## 💻 Запуск основных скриптов

### 1. Симуляция гонки в реальном времени:
```bash
python test_pipeline.py
```
*Запускает покруговой реплей Гран-при Бахрейна для Макса Ферстаппена с динамической реакцией на Safety Car, смену шин соперников и риски андерката.*

### 2. Сравнение пилотов Red Bull (VER vs PER):
```bash
python notebooks/red_bull_analysis.py
```
*Строит графики деградации шин и невязок модели для Ферстаппена и Переса на шинах Medium.*

### 3. Диагностика поведения модели для Фазы I:
```bash
python notebooks/phase1_visual_analytics.py
```
*Строит профиль телеметрии, график ошибок Residuals и таймлайн стратегии.*

---

## 📖 Руководство по выполнению Фазы I

Подробные инструкции по разделению на **AI-aided** и **Human-only** пары, правилам использования ИИ, исследовательским гипотезам и структуре 5-минутного видео описаны в файле:
👉 **[PHASE1_GUIDE.md](PHASE1_GUIDE.md)**

---

## 📅 Ключевые дедлайны курса

* **21.10.2026 (23:59):** Регистрация группы из 4 человек на портале курса.
* **28.10.2026 (23:59):** Регистрация задачи и модели (Task Selection).
* **11.11.2026 (23:59):** **Midpoint Submission (Фаза I)** — код обеих пар, AI-лог, 5-минутное видео.
* **09.12.2026 (23:59):** **Final Submission (Фаза II)** — полный код, итоговые логи, 10-минутная презентация.

import streamlit as st
import json
import numpy as np
import pandas as pd
import os
import gdown
from src.pipeline import DissonancePipeline
from src.config import Config
from transformers import pipeline as hf_pipeline # Добавили импорт

# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="Dissonance Analyzer", layout="wide")
st.title("Анализатор семантического диссонанса")
st.markdown("Система гибридного анализа публикаций Telegram на базе RuBERT и адаптивной векторизации реакций.")

# --- КЭШИРОВАНИЕ МОДЕЛЕЙ --- 
# https://drive.google.com/file/d/1Tds80YvaZlqAr2dnuPvgL-dnks_I-3Jx/view?usp=sharing
@st.cache_resource
def load_hybrid_pipeline():
    weights_path = "dissonance_model_1000.pth"
    file_id = '1Tds80YvaZlqAr2dnuPvgL-dnks_I-3Jx' 
    url = f'https://drive.google.com/uc?id={file_id}'

    if not os.path.exists(weights_path):
        with st.spinner('Загрузка весов модели из облака... Это может занять пару минут.'):
            try:
                gdown.download(url, weights_path, quiet=False)
            except Exception as e:
                st.error(f"Ошибка при скачивании весов: {e}")
    
    return DissonancePipeline(weights_path=weights_path)

@st.cache_resource
def load_sentiment_baseline():
    # Загружаем готовую классическую модель анализа тональности для сравнения
    return hf_pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")

pipeline = load_hybrid_pipeline()
sentiment_analyzer = load_sentiment_baseline()

# --- БОКОВОЕ МЕНЮ (НАСТРОЙКИ И АБЛЯЦИЯ) ---
st.sidebar.header("Параметры системы")

ablation_mode = st.sidebar.radio(
    "Режим анализа (Сравнение подходов):",
    ["Гибридный (Текст + Реакции)", "Классический (Только Текст)"],
    help="Сравнение предложенной гибридной системы с традиционным анализом тональности (Sentiment Analysis)."
)

tau_threshold = st.sidebar.slider(
    "Порог аномалии (τ)", min_value=0.05, max_value=0.95, value=0.56, step=0.01
)
gamma = st.sidebar.slider(
    "Коэффициент адаптации (γ)", min_value=0.0, max_value=3.0, value=1.0, step=0.1
)
noise_tolerance = st.sidebar.slider(
    "Толерантность к шуму", min_value=0.0, max_value=1.0, value=Config.MAX_UNKNOWN_RATIO, step=0.05
)

# --- ОСНОВНОЙ ИНТЕРФЕЙС ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Входные данные")
    input_text = st.text_area("Текст публикации:", value="Средняя зарплата в России за 10 лет выросла почти в 3 раза, сообщают экономисты. Показатель поднялся с 32 633 рублей в 2016 году до 100 360 рублей в 2025-м", height=150)
    input_reactions = st.text_input("Реакции аудитории (JSON):", value=
                                    '{"ReactionPaid()": 464, "❤": 50, "🤣": 1821, "🤡": 9170, "🤬": 27, "👎": 149}')
    input_views = st.number_input("Количество просмотров (для расчета ER):", min_value=1, value=163000, step=1000)

    with st.expander("Шаблон для ручного ввода (12 реакций)"):
        st.markdown("Скопируй этот JSON, вставь в поле выше и впиши свои цифры. Ненужные нули можно оставить.")
        # Генерируем словарь, где ключи - реакции из конфига, а значения - 0
        template_dict = {emoji: 0 for emoji in Config.TARGET_REACTIONS}
        # Превращаем в красивую JSON строку
        st.code(json.dumps(template_dict, ensure_ascii=False), language='json')

with col2:
    st.subheader("Распределение")
    try:
        raw_dict = json.loads(input_reactions)
        reactions_dict = {k.replace('\ufe0f', ''): v for k, v in raw_dict.items()}
        if reactions_dict:
            df_reactions = pd.DataFrame(list(reactions_dict.items()), columns=['Эмодзи', 'Количество'])
            st.bar_chart(df_reactions.set_index('Эмодзи'))
    except:
        st.warning("Некорректный JSON реакций.")

# --- КНОПКА ЗАПУСКА И ЛОГИКА ---
if st.button("Анализировать публикацию", type="primary"):
    try:
        raw_dict = json.loads(input_reactions)
        reactions_dict = {k.replace('\ufe0f', ''): v for k, v in raw_dict.items()}
        total = sum(reactions_dict.values())
        known = sum(reactions_dict.get(e, 0) for e in Config.TARGET_REACTIONS)
        unknown_ratio = (total - known) / total if total > 0 else 0
        
        if unknown_ratio > noise_tolerance:
            st.error(f"Пост забракован: доля неизвестных реакций ({unknown_ratio:.1%}) превышает порог ({noise_tolerance:.1%}).")
        else:
            st.markdown("---")
            st.subheader("Результаты анализа")
            
            if ablation_mode == "Классический (Только Текст)":
                st.info("Классический подход: модель игнорирует реакции и пытается оценить только тональность текста (как описано в Главе 1).")
                
                # Запускаем готовую модель
                result = sentiment_analyzer(input_text)[0]
                label_rus = {"POSITIVE": "Позитив", "NEGATIVE": "Негатив", "NEUTRAL": "Нейтрально"}.get(result['label'], result['label'])
                
                res_col1, res_col2 = st.columns(2)
                res_col1.metric("Тональность текста", label_rus)
                res_col2.metric("Уверенность модели", f"{result['score']:.1%}")
                
                if label_rus == "Негатив":
                    st.warning("Классическая модель видит только негативный текст и не способна распознать саркастичный смех аудитории.")
                
            else:
                raw_reactions_array = np.array([reactions_dict.get(e, 0) for e in Config.TARGET_REACTIONS])
                
                total_reactions = sum(reactions_dict.values())
                
                custom_stats = {
                    'n_total': Config.DEFAULT_CHANNEL_STATS['n_total'],
                    'n_ri_c': np.array([Config.DEFAULT_CHANNEL_STATS['n_ri_c'][e] for e in Config.TARGET_REACTIONS]),
                    'gamma': gamma
                }
                
                # Инференс гибридной модели
                result = pipeline.process_publication(input_text, raw_reactions_array, custom_stats)
                score = result['dissonance_score']
                is_anomaly = score >= tau_threshold

                er_percent = (total_reactions / input_views) * 100
                
                res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                
                res_col1.metric("Dissonance Score", f"{score:.4f}")
                res_col2.metric("Вовлеченность (ER)", f"{er_percent:.2f}%")
                res_col3.metric("Латентность", f"{result['latency_ms']} мс")
                
                if is_anomaly:
                    res_col3.error("АНОМАЛИЯ (Диссонанс)")
                else:
                    res_col3.success("НОРМА (Конгруэнтность)")
                    
                st.progress(score, text="Вероятность семантического диссонанса")

    except Exception as e:
        st.error(f"Произошла ошибка при обработке: {e}")
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Config:
    DB_PATH = os.path.join(BASE_DIR, 'telegram_dataset.db')
    MODEL_NAME = "DeepPavlov/rubert-base-cased"
    
    # Исходный список
    RAW_TARGET_REACTIONS = ['👍', '❤️', '🔥', '🤣', '🤯', '🤡', '🤬', '👎', '🤮', '💩', '😢', '😁']
    
    # Очищенный список, который будет использоваться во ВСЕМ коде
    # Мы удаляем невидимый селектор \ufe0f у каждого эмодзи
    TARGET_REACTIONS = [e.replace('\ufe0f', '') for e in RAW_TARGET_REACTIONS]

    # Максимальная доля неизвестных реакций (20%)
    MAX_UNKNOWN_RATIO = 0.20
    
    # Создаем сырой словарь
    _RAW_N_RI_C = {
        '👍': 500, '❤️': 300, '🔥': 200, '🤣': 150, '🤯': 50, '🤡': 40,
        '🤬': 30, '👎': 20, '🤮': 10, '💩': 5, '😢': 15, '😁': 100
    }


    DEFAULT_CHANNEL_STATS = {
        'n_total': 1000,
        'n_ri_c': {k.replace('\ufe0f', ''): v for k, v in _RAW_N_RI_C.items()},
        'gamma': 1.0 
    }
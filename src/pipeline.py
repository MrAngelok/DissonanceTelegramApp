import os
import torch
import time
from transformers import AutoTokenizer
from loguru import logger
from .vectorizer import ReactionVectorizer
from .model import DissonanceModel
from .config import Config

class DissonancePipeline:
    def __init__(self, weights_path="dissonance_model_1000.pth"):
        logger.info("Initializing Dissonance Pipeline...")
        
        # Указываем путь к кэшу явно, чтобы избежать OSError
        cache_dir = "/tmp/huggingface"
        os.makedirs(cache_dir, exist_ok=True)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_NAME, 
            cache_dir=cache_dir,
            local_files_only=False
        )
        
        # Инициализируем модель с правильным количеством реакций (12)
        self.model = DissonanceModel(model_name=Config.MODEL_NAME, k_reactions=len(Config.TARGET_REACTIONS))
        
        # --- ЗАГРУЗКА ОБУЧЕННЫХ ВЕСОВ ---
        try:
            # map_location='cpu' нужен для безопасного переноса весов с GPU Colab на твой локальный ПК
            self.model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
            logger.success("Обученные веса успешно загружены! Модель поумнела.")
        except FileNotFoundError:
            logger.warning("Файл весов не найден. Используются случайные веса.")
            
        self.model.eval() # Режим инференса
        self.vectorizer = ReactionVectorizer(categories_count=len(Config.TARGET_REACTIONS))

    def process_publication(self, text: str, raw_reactions: list, channel_stats: dict):
        start_time = time.perf_counter()

        # 1. Токенизация
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # 2. Векторизация реакций (O(1) сложность) 
        v_r_np = self.vectorizer.vectorize(
            counts=raw_reactions,
            n_total=channel_stats['n_total'],
            n_ri_c=channel_stats['n_ri_c'],
            gamma=channel_stats['gamma']
        )
        v_r = torch.tensor(v_r_np, dtype=torch.float32).unsqueeze(0)

        # 3. Инференс модели
        with torch.no_grad():
            raw_logit = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                v_r=v_r
            )
            probability = torch.sigmoid(raw_logit)
            
        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "dissonance_score": probability.item(),
            "latency_ms": round(latency_ms, 2)
        }
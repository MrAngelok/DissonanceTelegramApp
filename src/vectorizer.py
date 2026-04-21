import numpy as np
import unicodedata

class ReactionVectorizer:
    def __init__(self, categories_count=12):
        self.k = categories_count

    def vectorize(self, counts: np.ndarray, n_total: int, n_ri_c: np.ndarray, gamma: float) -> np.ndarray:
        """
        Реализация алгоритма адаптивной векторизации (формулы из 2.2 ВКР).
        """
        # Проверка условия на отсутствие реакций 
        if np.sum(counts) == 0:
            return np.zeros(self.k)

        # Вычисление базового веса (Inverse Reaction Frequency) 
        lambda_base = np.log(1 + n_total / (1 + n_ri_c))
        
        # Корректирующий множитель по типу канала (через степень)
        # Это предотвращает математическое сокращение при нормализации
        lambda_adj = lambda_base ** gamma
        
        # Поэлементное взвешивание
        w = counts * lambda_adj
        
        # Нормализация в вероятностное распределение 
        v_r = w / np.sum(w)
        
        return v_r
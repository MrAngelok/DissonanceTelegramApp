import torch
import torch.nn as nn
from transformers import AutoModel

class DissonanceModel(nn.Module):
    def __init__(self, model_name="DeepPavlov/rubert-base-cased", k_reactions=12):
        super().__init__()
        # Использование предобученной модели для русскоязычного сегмента 
        self.text_encoder = AutoModel.from_pretrained(model_name)
        
        # Заморозка весов BERT (опционально, для ускорения начальных тестов)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Легковесная MLP для конкатенированного вектора (768 + 12) 
        self.mlp = nn.Sequential(
            nn.Linear(768 + k_reactions, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1) # Проекция в скалярный логит 
        )

    def forward(self, input_ids, attention_mask, v_r):
        # Получение эмбеддинга текста (используем [CLS] токен)
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        v_t = outputs.last_hidden_state[:, 0, :] # Размерность: [batch_size, 768] 
        
        # Механизм слияния модальностей (Fusion Layer) 
        z = torch.cat((v_t, v_r), dim=1) 
        
        # Классификационная головка
        logit = self.mlp(z)
        probability = torch.sigmoid(logit) # Вероятность аномалии 
        
        return probability
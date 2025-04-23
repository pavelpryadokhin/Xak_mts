import torch
import numpy as np
from transformers import AutoTokenizer
import nltk
from nltk.corpus import stopwords
from natasha import MorphVocab, Doc, Segmenter, NewsEmbedding, NewsMorphTagger
import re
from transformers import BertForSequenceClassification
import sys
import os


class IntentPredictor:
    def __init__(self, model_path, labels_dict, max_length=128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-conversational")
        self.model = BertForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

        self.labels = labels_dict
        self.max_length = max_length

        # Инициализация компонентов предобработки
        nltk.download('stopwords', quiet=True)
        self.stopwords_ru = set(stopwords.words("russian"))
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.morph_tagger = NewsMorphTagger(NewsEmbedding())

    def _preprocess_text(self, text):
        """Метод предобработки текста"""
        # Очистка текста
        text = re.sub("[^а-яА-ЯёЁa-zA-Z0-9]", " ", text)
        text = re.sub(r'\s+', ' ', text).strip().lower()

        # Сегментация и лемматизация
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)

        # Извлечение лемм и фильтрация стоп-слов
        tokens = []
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
            lemma = token.lemma.lower()
            if lemma not in self.stopwords_ru and len(lemma) > 2:
                tokens.append(lemma)
        return ' '.join(tokens)

    def preprocess(self, text):
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return encoding['input_ids'].to(self.device), encoding['attention_mask'].to(self.device)

    def predict(self, text):
        # Предобработка
        text = self._preprocess_text(text)
        # Токенизация
        input_ids, attention_mask = self.preprocess(text)


        # Предсказание
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_idx = np.argmax(probabilities)

        return {
            'label': self.labels[predicted_idx],
            'confidence': float(probabilities[predicted_idx]),
            'db_categories': f'id_category_{predicted_idx}'
        }

db_categories = {
    0: "Финансовые и платежные услуги: карты, кредиты, переводы, страхование",
    1: "Мобильная связь, интернет и ТВ для дома и бизнеса",
    2: "Бизнес-решения и корпоративные услуги",
    3: "Инновационные технологии и IoT устройства",
    4: "Поддержка, контроль качества и тестирование",
    5: "Розничные и торговые платформы, партнерские сети"
}


# Пример использования
if __name__ == "__main__":
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'bert_model'))
    predictor = IntentPredictor(model_path=model_path,labels_dict=db_categories)
    text = "аи агенты какие есть"
    result = predictor.predict(text)

    print(f"Предсказанный класс: {result['label']}")
    print(f"Уверенность: {result['confidence']:.4f}")
    print(f"Категория базы данных: {result['db_categories']}")
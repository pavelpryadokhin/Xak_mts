from transformers import pipeline
import re

class EmotionPredictor:
    def __init__(self, model_name="cointegrated/rubert-tiny2-cedr-emotion-detection"):
        self.model_name = model_name
        self.pipe = pipeline("text-classification", model=model_name)
    
    def preprocess_text(self, text):
        # Очистка текста
        text = re.sub("[^а-яА-ЯёЁa-zA-Z0-9!,.?]", " ", text)
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text
    
    def predict(self, text):
        processed_text = self.preprocess_text(text)
        result = self.pipe(processed_text)
        return result


if __name__ == "__main__":
    text = 'У меня двойное, списание средств на карте!'
    predictor = EmotionPredictor()
    result = predictor.predict(text)
    print({'label':result[0]['label'],'score':result[0]['score']})
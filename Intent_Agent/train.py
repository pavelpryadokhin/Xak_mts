import time
import datetime
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer,  BertForSequenceClassification,  get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from natasha import MorphVocab, Doc, Segmenter, NewsEmbedding, NewsMorphTagger
import re
from bs4 import BeautifulSoup
import html
from tqdm import tqdm
import pandas as pd

class DataPreprocessor:
    def __init__(self, train_df, text_column='article_title', label_column='category'):
        self.train_df = train_df
        self.text_column = text_column
        self.label_column = label_column
        self.ohe = preprocessing.OneHotEncoder()
        self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-conversational")
        self.unique_labels = sorted(train_df[label_column].unique().tolist())

        # Инициализация компонентов для предобработки
        nltk.download('stopwords', quiet=True)
        self.stopwords_ru = set(stopwords.words("russian"))
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.morph_tagger = NewsMorphTagger(NewsEmbedding())

    def _preprocess_text(self, text):
        """Метод предобработки текста"""
        # Конвертация HTML в plain text
        # soup = BeautifulSoup(text, 'html.parser')
        # text = soup.get_text(separator=' ')
        # text = html.unescape(text)

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

    def preprocess_data(self, test_size=0.1, random_state=7):

        X = [self._preprocess_text(text) for text in tqdm(self.train_df[self.text_column].values)]
        y = self.train_df[self.label_column].values

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )

        return X_train, X_valid, y_train, y_valid

    def tokenize_data(self, texts, max_len=128):
        input_ids = []
        attention_masks = []

        for text in texts:
            encoded = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_len,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True
            )
            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])

        return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

class ModelTrainer:
    def __init__(self, model_name="DeepPavlov/rubert-base-cased-conversational", num_labels=6, device=None):
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False
        ).to(self.device)
        self.optimizer = None
        self.scheduler = None
        self.training_stats = []

    @staticmethod
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    @staticmethod
    def format_time(elapsed):
        return str(datetime.timedelta(seconds=int(round(elapsed))))

    def create_dataloaders(self, X_train, X_valid, y_train, y_valid, batch_size=32):
        train_ids, train_masks = X_train
        val_ids, val_masks = X_valid

        train_dataset = TensorDataset(train_ids, train_masks, torch.tensor(y_train))
        val_dataset = TensorDataset(val_ids, val_masks, torch.tensor(y_valid))

        train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=batch_size
        )

        val_dataloader = DataLoader(
            val_dataset,
            sampler=SequentialSampler(val_dataset),
            batch_size=batch_size
        )

        return train_dataloader, val_dataloader

    def initialize_optimizer(self, train_dataloader, epochs=4, learning_rate=2e-5, eps=1e-8):
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=eps)
        total_steps = len(train_dataloader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

    def train_epoch(self, train_dataloader):
        total_train_loss = 0
        self.model.train()

        for batch in train_dataloader:
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            self.optimizer.zero_grad()

            output = self.model(
                b_input_ids,
                attention_mask=b_input_mask,
                labels=b_labels
            )

            loss = output.loss
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

        return total_train_loss / len(train_dataloader)

    def evaluate(self, val_dataloader):
        self.model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0

        for batch in val_dataloader:
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            with torch.no_grad():
                output = self.model(
                    b_input_ids,
                    attention_mask=b_input_mask,
                    labels=b_labels
                )

            loss = output.loss
            logits = output.logits

            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += self.flat_accuracy(logits, label_ids)

        return (
            total_eval_loss / len(val_dataloader),
            total_eval_accuracy / len(val_dataloader)
        )

    def train(self, train_dataloader, val_dataloader, epochs=4, save_path='bert_model'):
        best_accuracy = 0
        total_t0 = time.time()

        for epoch_i in range(epochs):
            print(f"\n======== Epoch {epoch_i + 1} / {epochs} ========")

            # Training
            t0 = time.time()
            avg_train_loss = self.train_epoch(train_dataloader)
            training_time = self.format_time(time.time() - t0)

            # Validation
            t0 = time.time()
            avg_val_loss, avg_val_accuracy = self.evaluate(val_dataloader)
            validation_time = self.format_time(time.time() - t0)

            # Save best model
            if avg_val_accuracy > best_accuracy:
                self.model.save_pretrained(save_path)
                best_accuracy = avg_val_accuracy

            # Record statistics
            self.training_stats.append({
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            })

            print(f"\nAverage training loss: {avg_train_loss:.2f}")
            print(f"Validation Accuracy: {avg_val_accuracy:.2f}")
            print(f"Epoch time: {training_time} training, {validation_time} validation")

        print(f"\nTotal training took {self.format_time(time.time()-total_t0)}")


if __name__ == "__main__":
    df=pd.read_csv('Data/parse.csv')
    # # # Инициализация и предобработка данных
    preprocessor = DataPreprocessor(df)
    X_train, X_valid, y_train, y_valid = preprocessor.preprocess_data()

    # Токенизация
    train_ids, train_masks = preprocessor.tokenize_data(X_train)
    val_ids, val_masks = preprocessor.tokenize_data(X_valid)


    # Инициализация тренера
    trainer = ModelTrainer()
    train_dataloader, val_dataloader = trainer.create_dataloaders(
        (train_ids, train_masks),
        (val_ids, val_masks),
        y_train,
        y_valid,
        batch_size=32
    )

    # Настройка оптимизатора и обучение
    epochs=4
    trainer.initialize_optimizer(train_dataloader,epochs=epochs)
    trained_model = trainer.train(train_dataloader, val_dataloader,epochs=epochs)

'''
Обучение:
======== Epoch 1 / 4 ========

Average training loss: 1.47
Validation Accuracy: 0.65
Epoch time: 0:01:04 training, 0:00:02 validation

======== Epoch 2 / 4 ========

Average training loss: 0.82
Validation Accuracy: 0.78
Epoch time: 0:01:04 training, 0:00:02 validation

======== Epoch 3 / 4 ========

Average training loss: 0.49
Validation Accuracy: 0.85
Epoch time: 0:01:03 training, 0:00:02 validation

======== Epoch 4 / 4 ========

Average training loss: 0.33
Validation Accuracy: 0.84
Epoch time: 0:01:04 training, 0:00:02 validation

Total training took 0:05:00
'''
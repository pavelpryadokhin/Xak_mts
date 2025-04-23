from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import Dict, List
import pandas as pd
from qdrant_client import QdrantClient
from langchain_community.vectorstores.qdrant import Qdrant
import sys
import os
from dotenv import load_dotenv
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import CustomAPIEmbeddings

dotenv_path = Path(__file__).parents[1] / '.env'
load_dotenv(dotenv_path=dotenv_path)

class QdrantDBManager:
    def __init__(self, embedding_model, qdrant_client):
        self.embedding_model = embedding_model
        self.qdrant_client = qdrant_client
        self.qdrant_host = qdrant_client._client._host
        self.qdrant_port = qdrant_client._client._port
        self.max_chunk_size = 1000
        self.headers_to_split = [("#", "H1"), ("##", "H2"), ("###", "H3")]

    def _process_dataframe(self, df: pd.DataFrame) -> List[Document]:
        documents = []
        splitter_markdown = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split,
            strip_headers=True
        )
        splitter_recursive = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=20,
            separators=["\n"]
        )

        for _, row in df.iterrows():
            question = f"{row['article_title']}\n"
            
            try:
                # Первичное разделение по Markdown-заголовкам
                header_chunks = splitter_markdown.split_text(row['content_markdown'])
                
                for chunk in header_chunks:
                    # Добавляем заголовки из метаданных
                    headers = '\n'.join(chunk.metadata.values()) + '\n'
                    full_content = question+headers + chunk.page_content
                    # Дополнительное разделение длинных фрагментов
                    if len(full_content) > self.max_chunk_size:
                        for sub in  splitter_recursive.split_text(chunk.page_content):
                            documents.append(Document(
                                page_content=question+headers +sub,
                                metadata=chunk.metadata
                            ))
                    else:
                        documents.append(Document(
                            page_content=full_content,
                            metadata=chunk.metadata
                        ))
            except Exception as e:
                print(f"Ошибка при обработке строки {_}: {str(e)}")
                continue
                
        return documents

    def create_collection(self, df: pd.DataFrame, db_name: str) -> None:
        documents = self._process_dataframe(df)
        print(f"Подготовлено {len(documents)} документов для добавления в коллекцию")
        
        try:
            Qdrant.from_documents(
                documents=documents,
                embedding=self.embedding_model,
                host=self.qdrant_host,
                port=self.qdrant_port,
                collection_name=db_name,
                force_recreate=True
            )
            print(f"Коллекция {db_name} успешно создана")
        except Exception as e:
            print(f"Ошибка при создании коллекции: {str(e)}")

if __name__ == "__main__":
    df=pd.read_csv('Data/parse.csv')
    categories=df['category'].unique()
    dfs_by_category = {
    category: df[df['category'] == category].drop(columns=['category','content'])
    for category in categories
    }
    embeddings = CustomAPIEmbeddings(
        api_url=os.getenv("EMBEDDING_API_URL"),
        api_key=os.getenv("EMBEDDING_API_KEY"),
        model=os.getenv("EMBEDDING_MODEL"),
    )
    qdrant_client = QdrantClient(
        host=os.getenv("QDRANT_HOST"), 
        port=int(os.getenv("QDRANT_PORT"))
    )
    db_manager = QdrantDBManager(embeddings, qdrant_client)
    for i in range(len(dfs_by_category)):
        db_manager.create_collection(dfs_by_category[i], f'id_category_{i}')



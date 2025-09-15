from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from pinecone import Pinecone as PineconeClient, ServerlessSpec
import os
from dotenv import load_dotenv

class ChatBot:
    def __init__(self):
        load_dotenv()
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_TOKEN")

        # Initialize Pinecone
        pc = PineconeClient(api_key=os.getenv("PINECONE_TOKEN"))
        index_name = "mental-health-bot"
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="gcp", region="us-central1")
            )

        # Load data
        loader = TextLoader("depression_resources.txt")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        
        # Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Create Pinecone index
        self.docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
        
        # LLM
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.7, "max_length": 512})
        
        # Prompt template for mental health
        template = """
        You are a symptom tracking chatbot for mental health. Converse with the user only about mental health topics.
        Collect symptoms during the conversation. When you have enough info or user says "Thank you, I'm done", return a list of symptoms and a possible mental health issue.
        Past messages: {pasts}
        Context: {context}
        Question: {question}
        Answer:
        """
        self.prompt = PromptTemplate(template=template, input_variables=["pasts", "context", "question"])
        
        # Chain RAG
        self.rag_chain = (
            {"context": self.docsearch.as_retriever(), "question": RunnablePassthrough(), "pasts": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
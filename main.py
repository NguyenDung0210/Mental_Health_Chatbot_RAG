from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from pinecone import Pinecone, ServerlessSpec
import os
import time
from dotenv import load_dotenv

class ChatBot:
    def __init__(self):
        print("Starting ChatBot initialization...")
        # Load environment variables
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not hf_token or not pinecone_api_key:
            raise ValueError("HF_TOKEN or PINECONE_API_KEY not found in .env")
        print("Environment variables loaded.")

        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
        
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        index_name = "mental-health-bot"
        print(f"Checking Pinecone index: {index_name}")
        if index_name not in pc.list_indexes().names():
            print(f"Creating Pinecone index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                deletion_protection="disabled"
            )
            print("Index created. Waiting for it to be ready...")
            for _ in range(30):  # Chờ tối đa 3 phút
                index_desc = pc.describe_index(index_name)
                if index_desc.get("status", {}).get("ready", False):
                    print("Index is ready!")
                    break
                time.sleep(6)
            else:
                raise RuntimeError(f"Index {index_name} not ready after 3 minutes.")
        print("Pinecone initialized.")

        # Load data
        print("Loading documents...")
        try:
            loader = TextLoader("depression_resources.txt")
            documents = loader.load()
            print(f"Loaded {len(documents)} documents.")
        except Exception as e:
            raise RuntimeError(f"Failed to load documents: {str(e)}")

        text_splitter = CharacterTextSplitter(chunk_size=6000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        print(f"Split into {len(docs)} chunks.")

        # Embeddings
        print("Initializing embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("Embeddings initialized.")

        # Create Pinecone vector store
        print("Creating Pinecone vector store...")
        try:
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            existing_vectors = stats.get("total_vector_count", 0)
            print(f"Index has {existing_vectors} vectors.")
            if existing_vectors >= len(docs):
                print("Vectors already exist. Using existing index...")
                self.docsearch = PineconeVectorStore(
                    index_name=index_name,
                    embedding=embeddings,
                    pinecone_api_key=pinecone_api_key
                )
            else:
                print("Upserting new vectors...")
                self.docsearch = PineconeVectorStore.from_documents(
                    documents=docs,
                    embedding=embeddings,
                    index_name=index_name,
                    pinecone_api_key=pinecone_api_key
                )
            print("Pinecone vector store created.")
        except Exception as e:
            raise RuntimeError(f"Failed to create Pinecone vector store: {str(e)}")
        
        # LLM
        print("Initializing LLM...")
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            task="text-generation",
            huggingfacehub_api_token=hf_token,
            temperature=0.7,
            model_kwargs={"max_new_tokens": 512},  # Thay max_length bằng max_new_tokens
            timeout=30
        )
        print("LLM initialized.")
        
        # Prompt template
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
        print("Setting up RAG chain...")
        self.rag_chain = (
            {
                "context": lambda x: self.docsearch.as_retriever().invoke(x["question"]),
                "question": lambda x: x["question"],
                "pasts": lambda x: x["pasts"]
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        print("ChatBot initialization completed.")

if __name__ == "__main__":
    print("Running main.py...")
    chatbot = ChatBot()
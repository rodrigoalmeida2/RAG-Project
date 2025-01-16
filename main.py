from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

login(token="seu token")

# Configuração do modelo LLaMA
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Base de conhecimento
knowledge_base = [
    "Hello World é usado em exemplos iniciais de programação.",
    "RAG significa Retrieval-Augmented Generation.",
    "FAISS é uma biblioteca para busca eficiente de vetores.",
    "LLaMA é um modelo avançado de linguagem desenvolvido pela Meta."
]

# Criando embeddings com HuggingFace
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Indexando a base de conhecimento no FAISS
vector_store = FAISS.from_texts(knowledge_base, embedding_model)

# Função de recuperação de informações
def retrieve_information(query):
    docs = vector_store.similarity_search(query, k=1)  # Retorna o documento mais relevante
    return docs[0].page_content if docs else "Nenhuma informação relevante encontrada."

# Função para gerar texto com o LLaMA
def generate_response(query):
    retrieved_info = retrieve_information(query)
    prompt = f"Baseado nas informações recuperadas: '{retrieved_info}', responda: {query}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Consulta
query = "O que é Hello World?"
response = generate_response(query)
print(response)

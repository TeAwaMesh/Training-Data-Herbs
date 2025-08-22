from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
import pyttsx3

# 1. Load the HTML Herbal file
with open("The Complete Herbal _ Project Gutenberg.html", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")
    text = soup.get_text()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([text])

# 3. Embed and store in Chroma
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma.from_documents(chunks, embeddings)

# 4. Load local LLM (Mistral via Ollama)
llm = Ollama(model="mistral")

# 5. Create Retrieval-QA chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# 6. Setup TTS
engine = pyttsx3.init()

def ask(query, speak=False):
    response = qa.run(query)
    print("\n🔮 Answer:", response, "\n")
    if speak:
        engine.say(response)
        engine.runAndWait()

# Interactive loop
while True:
    q = input("Ask about herbs (or type 'quit'): ")
    if q.lower() == "quit":
        break
    ask(q, speak=True)

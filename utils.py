from llama_cpp import Llama
import random
from transformers import MarianMTModel, MarianTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
import os
import asyncio
import aiofiles  # Asynchronous file handling
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import Document
import json
import re
import html
import hashlib
from langdetect import detect

model_cache = {"llama": None, "translation": {}, "embedding": None}


def load_llama_model(path="./bitnet_b1_58-large.Q4_0.gguf"):
    if not model_cache["llama"]:
        model_cache["llama"] = Llama(model_path=path, use_cache=False)
    return model_cache["llama"]

# Load translation model once
def load_translation_model(src_lang, tgt_lang):
    model_key = f"{src_lang}-{tgt_lang}"
    if model_key not in model_cache["translation"]:
        model_path = f"./local_models/opus-mt-{src_lang}-{tgt_lang}_model"
        tokenizer_path = f"./local_models/opus-mt-{src_lang}-{tgt_lang}_tokenizer"
        model_cache["translation"][model_key] = {
            "model": MarianMTModel.from_pretrained(model_path),
            "tokenizer": MarianTokenizer.from_pretrained(tokenizer_path),
        }
    return model_cache["translation"][model_key]

# Decode HTML entities
def decode_html_entities(text):
    return html.unescape(text)

# Translate text
def translate(text, src_lang, tgt_lang):
    try:
        model_data = load_translation_model(src_lang, tgt_lang)
        tokenizer, model = model_data["tokenizer"], model_data["model"]

        if isinstance(text, list):
            text = " ".join(text)
        elif not isinstance(text, str):
            raise ValueError("Input text must be a string or list of strings.")

        text = decode_html_entities(text)
        text = re.sub(r'<[^>]*>', '', text)

        input_ids = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translations = model.generate(**input_ids)

        translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translations]
        clean_translated_text = " ".join(decode_html_entities(translated_text) for translated_text in translated_texts)
        clean_translated_text = re.sub(r'<[^>]*>', '', clean_translated_text)

        return clean_translated_text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

# Detect language
def detect_language(text):
    lang = detect(text)  
    if lang in ['en', 'id']:
        return lang
    indonesian_vocab = ["halo", "apa", "nama", "berapa", "buat", "contoh", "apakah", "maksud", "jelaskan", "bagaimana", "tolong", "bisa", "arti", "cara", "dimana", "kenapa", "siapa", "beri", "itu", "makan", "minum", "ceritakan", "bantu", "dengan"
    ]
    english_vocab = ["hello", "at", "how", "are", "you", "please", "yes", "no", "where", "why", "who", "what", "how", "can", "help", "me", "morning", "evening", "sorry", "i", "from", "does", "explain", "solve", "the", "meaning", "tell", "a"
    ] 
    chunks = re.split(r'[.?!,;]', text)
    indonesian_matches = sum(1 for chunk in chunks for word in chunk.split() if word in indonesian_vocab)
    english_matches = sum(1 for chunk in chunks for word in chunk.split() if word in english_vocab)

    if indonesian_matches >= 2:
        return 'id'
    if english_matches >= 2:
        return 'en'
    
    return 'unknown'

def hash_document(doc):
    """Generate a hash for the document content."""
    content = str(doc.page_content) + str(doc.metadata)
    doc_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
    return doc_hash

# Load and split documents asynchronously
async def load_document(file_path):
    documents = []
    try:
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()

            language = detect_language(content)

            if file_path.endswith(".json"):
                data = json.loads(content)
                for item in data:
                    text_content = str(item.get("isi", "")).strip()  # Ensure string type
                    if text_content:  # Only process non-empty strings
                        doc = Document(page_content=text_content, metadata={"file": file_path, "language": language})
                        doc.metadata["hash"] = hash_document(doc)
                        documents.append(doc)
                    else:
                        print(f"Skipping empty or invalid content in JSON item: {item}")
            else:
                text_content = content.strip()
                if text_content:  # Only process non-empty strings
                    doc = Document(page_content=text_content, metadata={"file": file_path, "language": language})
                    doc.metadata["hash"] = hash_document(doc)
                    documents.append(doc)
                else:
                    print(f"Skipping empty file: {file_path}")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return [doc for doc in documents if doc.page_content]  # Ensure valid content

async def load_documents_async(file_paths):
    tasks = [asyncio.create_task(load_document(file_path)) for file_path in file_paths]
    documents = await asyncio.gather(*tasks)
    return [doc for sublist in documents for doc in sublist]

# Split documents into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30, add_start_index=True)
    split_docs = text_splitter.split_documents(documents)
    return [doc for doc in split_docs if doc.page_content]

# Process documents concurrently
def process_documents(folder_path):
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith((".json", ".txt"))]
    raw_documents = asyncio.run(load_documents_async(file_paths))
    return split_documents(raw_documents)

# Set up the vector store
def setup_vector_store(embedding_model, persist_dir="./chroma_db"):
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
   
    # # Filter documents by language if specified
    # if language_filter:
    #     docs = [doc for doc in docs if doc.metadata.get("language") == language_filter]
    #     print(f"Filtering documents by language: {language_filter}. Remaining documents: {len(docs)}")

    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

def remove_repetitions(output):
    words = output.split()
    unique_words = []
    for word in words:
        if word not in unique_words:
            unique_words.append(word)
    return " ".join(unique_words)

# Filter relevant context
def filter_relevant_context(contexts, prompt, embedding_model, max_contexts=3, threshold=0.6):
    try:
        prompt_embedding = embedding_model.embed_query(prompt)
        similarities = [
            (ctx, cosine_similarity([prompt_embedding], [embedding_model.embed_query(ctx)])[0][0])
            for ctx in contexts
        ]
        filtered_similarities = [
            (ctx, score) for ctx, score in similarities if score >= threshold
        ]
        # Sort and return the top contexts
        sorted_similarities = sorted(filtered_similarities, key=lambda x: x[1], reverse=True)
        top_contexts = [ctx for ctx, _ in sorted_similarities[:max_contexts]]
        return top_contexts
    except Exception as e:
        print(f"Error in filter_relevant_context: {e}")
        return []

def clean_contexts(contexts):
    """
    Remove duplicates, excessive whitespace, and noise from contexts.
    """
    seen = set()
    cleaned = []
    for ctx in contexts:
        ctx_cleaned = " ".join(ctx.split())  # Remove extra whitespace
        if ctx_cleaned not in seen:  # Avoid duplicates
            cleaned.append(ctx_cleaned)
            seen.add(ctx_cleaned)
    return cleaned

# Condense output
def condense_output(output, max_sentences=2):
    """
    Condense the output to meaningful summary, limited to max_sentences.
    """
    sentences = re.split(r'(?<=[.!?])\s+', output)  # Split by sentence-ending punctuation
    condensed = " ".join(sentences[:max_sentences]).strip()
    return condensed if condensed else "I'm sorry, I don't understand your request."

# Process the user prompt
def process_prompt(prompt, folder_path, model, embedding_path="./local_models/all-MiniLM-L6-v2_embeddings"):
    if not model_cache["embedding"]:
        model_cache["embedding"] = HuggingFaceEmbeddings(model_name=embedding_path)
 
    language = detect_language(prompt)
    folder_path = folder_path + ("id/" if language == "id" else "en/")

    docs = process_documents(folder_path)
    docs = [doc for doc in docs if doc.metadata.get("language") == language]

    retriever = setup_vector_store(model_cache["embedding"])

    context = retriever.invoke(prompt)
    print(context)
    context_texts = [doc.page_content for doc in context]
    cleaned_contexts = clean_contexts(context_texts)
    relevant_contexts = filter_relevant_context(cleaned_contexts, prompt, model_cache["embedding"])
 
    if not relevant_contexts:
        relevant_contexts = ["Sorry, I couldn't find relevant context."]
    print("\nRelevan Context: ",relevant_contexts)

    if language == "id":
        prompt = translate(prompt, "id", "en")
        relevant_contexts = [translate(ctx, "id", "en") if isinstance(ctx, str) else ctx for ctx in relevant_contexts]

    template = """
Use the context below to generate an appropriate response.

Context:
{context}

Question: {question}
"""
    custom_rag_prompt = PromptTemplate.from_template(template)
    final_prompt = custom_rag_prompt.format(context="\n\n".join(relevant_contexts), question=prompt)
    print("\nFinal Prompt: ",final_prompt)

    response = model(
        prompt=final_prompt,
        max_tokens=512,
        temperature=random.uniform(0.6, 1.0),
        top_p=0.95,
        top_k=40,
    )
    output = response["choices"][0]["text"].strip()
    output = remove_repetitions(output)
    output = condense_output(output)
    return translate(output, "en", "id") if language == "id" else output


  

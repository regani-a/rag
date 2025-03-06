import random
import re
from llama_cpp import Llama
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect
from concurrent.futures import ThreadPoolExecutor
import html


model_cache = {"llama": None, "translation": {}}

def load_llama_model(path="./bitnet_b1_58-large.Q4_0.gguf"):
    if not model_cache["llama"]:
        model_cache["llama"] = Llama(model_path=path)
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

def decode_html_entities(text):
    return html.unescape(text)

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

def generate_response(model, prompt, temperature, top_p, max_tokens):
    """Generates a response from the Llama model."""
    response = model(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=40
    )
    print("TEMPERATURE:" ,temperature)
    print("TOP_P:" ,top_p)
    return response["choices"][0]["text"].strip()

def process_prompt(prompt, max_tokens, temperature, top_p):
    """Processes the input prompt by checking the language and translating if needed."""
    if model_cache["llama"] is None:
            model_cache["llama"] = load_llama_model()
    language = detect_language(prompt)
    try:
        if language == 'id':
            prompt_with_instruction_id = f"Jelaskan atau jawab pertanyaan ini secara singkat: {prompt}"

            # Translate Indonesian to English
            translated_input = translate(prompt_with_instruction_id, 'id', 'en')
            print(f"Translated Input (ID -> EN): {translated_input}")
            
            # Generate response in English
            output_text_en = generate_response(model_cache["llama"], translated_input, temperature, top_p, max_tokens)
            
            # Translate back to Indonesian
            output_text = translate(output_text_en, 'en', 'id')
        else:
            prompt_with_instruction_en = f"Explain or answer briefly this: {prompt}"

            # Process English directly
            output_text = generate_response(model_cache["llama"], prompt_with_instruction_en, temperature, top_p, max_tokens)

        # Post-process to clean unnecessary repetition
        return clean_repetitions(output_text)
    finally:
        if model_cache["llama"]:
            model_cache["llama"].close()
        del model_cache["llama"]

from collections import Counter

def clean_repetitions(text):
    """
    Cleans text dynamically by removing repetitive words, phrases, and duplicate sentences.
    Works for any input, ensuring clean and logical output.
    """
    # Step 1: Remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)

    # Step 2: Split text into sentences and deduplicate
    sentences = text.split(". ")
    unique_sentences = list(dict.fromkeys(sentences))  # Removes duplicate sentences

    # Step 3: Detect and clean frequent word repetitions dynamically
    word_list = " ".join(unique_sentences).split()
    word_counts = Counter(word_list)

    # Identify words repeated excessively (threshold: > 3 times)
    repeated_words = {word for word, count in word_counts.items() if count > 3}

    # Remove excessively repeated words
    cleaned_sentences = []
    for sentence in unique_sentences:
        words = sentence.split()
        cleaned_words = [word for word in words if word not in repeated_words]
        cleaned_sentences.append(" ".join(cleaned_words))

    # Step 4: Reconstruct cleaned text
    cleaned_text = ". ".join([s.strip() for s in cleaned_sentences if s.strip()])

    # Step 5: Clean unnecessary spaces or punctuation
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    cleaned_text = re.sub(r'\s\.', '.', cleaned_text)  # Fix space before periods
    
    return cleaned_text


# Example prompt
prompt = "apa itu Data"
input_token_length = len(prompt)

# Randomize parameters
temperature = random.uniform(0.6, 1.0)
top_p = random.uniform(0.8, 1.0)

# Determine token length and set max tokens
max_tokens = 200 if input_token_length < 50 else 150 if input_token_length < 100 else 100

# Using ThreadPoolExecutor for parallel processing
with ThreadPoolExecutor(max_workers=4) as executor:
    future = executor.submit(process_prompt, prompt, max_tokens, temperature, top_p)
    result = future.result()

result_cleaned = decode_html_entities(result)
result_cleaned = re.sub(r'<[^>]*>', '', result_cleaned)

print(f"Input: {prompt}")
print(f"Output: {result_cleaned}")

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from faster_whisper import WhisperModel
import fasttext
from ui import create_ui  
from languages_code import language_code_mapping  

# Load Faster Whisper model with large size, using GPU if available
model_whisper = WhisperModel("large", device="cuda", compute_type="float16")

# Load FastText language detection model
# C:\Users\Zuber Shaikh/OneDrive/Desktop/multi/Asset/model
fasttext_model_path = "C:/Users/Zuber Shaikh/OneDrive/Desktop/multi/Asset/model/lid.176.bin"  # Update with your actual path
fasttext_model = fasttext.load_model(fasttext_model_path)

# Load NLLB200 model and tokenizer
nllb_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
nllb_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

# Set up the translation pipeline
def get_translator(src_lang, tgt_lang):
    translator = pipeline('translation', model=nllb_model, tokenizer=nllb_tokenizer, 
                          src_lang=src_lang, tgt_lang=tgt_lang, max_length=400)
    return translator

# Function to transcribe audio using Faster Whisper
def transcribe_audio(audio):
    if not audio:
        return "No audio file provided", "Detection: Detecting..."
    
    segments, info = model_whisper.transcribe(audio, beam_size=5)
    transcription = ""
    for segment in segments:
        transcription += f"{segment.text} "
    return transcription, "Detection: Detecting..."

# Language detection function using FastText
def detect_language_fn(text):
    predictions = fasttext_model.predict(text)
    detected_lang = predictions[0][0].replace("__label__", "")
    
    detected_lang_name = [
    key for key, value in language_code_mapping.items() 
    if value.get('fasttext') == detected_lang
]

    
    # detected_lang_name = [key for key, value in language_code_mapping.items() if value['fasttext'] == detected_lang]
    
    if detected_lang_name:
        return detected_lang_name[0]
    else:
        return detected_lang

# NLLB translation function using Hugging Face pipeline
def translate_nllb(text, target_language):
    # Detect the source language
    detected_lang_name = detect_language_fn(text)
    src_nllb_code = language_code_mapping.get(detected_lang_name, {}).get('nllb')
    tgt_nllb_code = language_code_mapping.get(target_language, {}).get('nllb')
    
    if not src_nllb_code or not tgt_nllb_code:
        return "Source or target language not supported for NLLB translation."
    
    # Use the pipeline for translation
    translator = get_translator(src_nllb_code, tgt_nllb_code)
    translated_text = translator(text)[0]['translation_text']
    
    return translated_text

# Function to handle transcription and detection
def handle_transcription_and_detection(audio):
    transcription, _ = transcribe_audio(audio)
    
    if transcription.strip():
        detected_language = detect_language_fn(transcription)
        return transcription, f"Detection: {detected_language}"
    else:
        return "No transcription available", "Detection: Detecting..."

# Function to handle translation
# Function to handle translation
def handle_translation(transcription, target_language):
    if transcription.strip():
        # Detect the source language
        detected_lang_name = detect_language_fn(transcription)
        
        # If the detected language is the same as the target language, return the transcription as is
        if detected_lang_name == target_language:
            return transcription  # No need to translate
        
        # Proceed with translation if languages are different
        translated_text = translate_nllb(transcription, target_language)
        return translated_text
    else:
        return "No transcription available"


# Main function to run the Gradio app
if __name__ == "__main__":
    ui = create_ui(handle_transcription_and_detection, handle_translation)  # Pass logic functions to UI
    ui.launch(share=True)

import gradio as gr
import fasttext
import torch
from transformers import pipeline
from languages_code import language_code_mapping

# Initialize input language with a detection placeholder
INITIAL_LANG_DETECTION = "Detecting Language..."

# Load FastText model for language detection
model_path = "C:/Users/Zuber Shaikh/OneDrive/Desktop/multi/Asset/model/lid.176.bin"
fasttext_model = fasttext.load_model(model_path)

# Set device: 0 for GPU if available, -1 for CPU
device = 0 if torch.cuda.is_available() else -1

# Create translation pipeline with src_lang and tgt_lang support
translator = pipeline(
    task="translation",
    model="facebook/nllb-200-distilled-600M",
    device=device
)

# Define the available language codes from the mapping
LANGUAGE_CODES_FASTTEXT = {k: v['fasttext'] for k, v in language_code_mapping.items() if 'fasttext' in v}
LANGUAGE_CODES_NLLB = {k: v['nllb'] for k, v in language_code_mapping.items()}
LANGUAGE_CHOICES = list(LANGUAGE_CODES_FASTTEXT.keys())

# Real-time detection using FastText
def detect_language(text, user_selected_language, prev_text):
    if text != prev_text:
        # Text has changed, reset user_selected_language
        user_selected_language = False
        
    if len(text.strip()) == 0:
        # Return initial detection placeholder if no text
        return (gr.update(choices=[INITIAL_LANG_DETECTION] + LANGUAGE_CHOICES, 
                           value=INITIAL_LANG_DETECTION), 
                False,  # Reset user_selected_language
                text)
    
    if user_selected_language:
        # User has selected a language manually, do not update the dropdown
        return gr.no_update, user_selected_language, text
    else:
        if len(text.strip()) == 0:
            # Return initial detection placeholder if no text
            return gr.update(choices=[INITIAL_LANG_DETECTION] + LANGUAGE_CHOICES, value=INITIAL_LANG_DETECTION), user_selected_language, text
        # Detect language using FastText
        predictions = fasttext_model.predict(text.lower())
        detected_lang_code = predictions[0][0].split("__label__")[1]
        # Match detected language code to a language in LANGUAGE_CODES_FASTTEXT
        detected_lang = next((lang for lang, code in LANGUAGE_CODES_FASTTEXT.items() if detected_lang_code == code), "Unknown")
        # Prepare the updated dropdown options with detected language at the top
        input_languages = [f"Detected: {detected_lang}"] + LANGUAGE_CHOICES
        return gr.update(choices=input_languages, value=f"Detected: {detected_lang}"), user_selected_language, text

def translate_text(input_text, input_lang, target_lang):
    # Map input_lang and target_lang to NLLB language codes
    if input_lang.startswith("Detected: "):
        input_lang = input_lang[len("Detected: "):]

    # Get the NLLB language codes
    input_lang_code = LANGUAGE_CODES_NLLB.get(input_lang)
    target_lang_code = LANGUAGE_CODES_NLLB.get(target_lang)

    if not input_lang_code or not target_lang_code:
        return "Unsupported language selected."
    
    # Check if the input and target languages are the same
    if input_lang == target_lang:
        return input_text  # Return the input text as-is without translation

    # Translate using src_lang and tgt_lang parameters
    translated = translator(
        input_text,
        src_lang=input_lang_code,
        tgt_lang=target_lang_code
    )
    translation_text = translated[0]['translation_text']
    
    print(f"Translated '{input_text}' from {input_lang_code} to {target_lang_code}: {translation_text}")
    
    return translation_text

# Gradio Interface
def create_interface():
    # Use the same Blocks structure but return it instead of launching
    with gr.Blocks() as demo:
        gr.Markdown("""<h2 style='text-align: center;'>Text-to-Text Translation</h2>""")
        # State variables to keep track of user selection and previous text
        user_selected_language = gr.State(False)
        prev_text = gr.State('')

        with gr.Row():
            with gr.Column():
                input_language = gr.Dropdown(
                    choices=[INITIAL_LANG_DETECTION] + LANGUAGE_CHOICES,
                    label="Input Language",
                    value=INITIAL_LANG_DETECTION
                )
                text_input = gr.Textbox(
                    placeholder="Type text for translation...",
                    label="Input Text"
                )
            with gr.Column():
                target_language = gr.Dropdown(
                    choices=LANGUAGE_CHOICES,
                    label="Target Language",
                    value=LANGUAGE_CHOICES[0]
                )
                translated_text = gr.Textbox(label="Translated Text")

        translate_button = gr.Button("Translate")

        text_input.change(
            fn=detect_language,
            inputs=[text_input, user_selected_language, prev_text],
            outputs=[input_language, user_selected_language, prev_text]
        )

        input_language.change(
            fn=lambda selection: True,
            inputs=input_language,
            outputs=user_selected_language
        )

        translate_button.click(
            fn=translate_text,
            inputs=[text_input, input_language, target_language],
            outputs=translated_text
        )
    return demo

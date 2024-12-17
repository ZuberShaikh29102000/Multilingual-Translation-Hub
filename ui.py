import gradio as gr
from languages_code import language_code_mapping

def create_ui(handle_transcription_and_detection, handle_translation):
    INITIAL_LANG_DETECTION = "Detection: Detecting..."
    
    # Filter languages with a 'whisper' key
    whisper_languages = [lang for lang, codes in language_code_mapping.items() if 'whisper' in codes]
    language_list = [INITIAL_LANG_DETECTION, *whisper_languages]
    # language_list = ["Detection: Detecting...", *language_code_mapping.keys()]
    
    # with gr.Blocks() as interface:
        # Title and Sub-heading
    gr.Markdown("""<h2 style='text-align: center;'>Speech-to-Text Translation</h2>""")
        
        # Layout for Source Language and Target Language dropdowns side by side
    with gr.Row():
            source_language = gr.Dropdown(choices=language_list, label="Input Language", value=INITIAL_LANG_DETECTION, allow_custom_value=True)
            target_language = gr.Dropdown(choices=language_list[1:], label="Target Language")
        
        # Layout for audio_input and translation_output side by side
    with gr.Row():
            # Audio input on the left side
            with gr.Column():
                audio_input = gr.Audio(type="filepath", label="Record or Upload Audio for Transcription")
                transcription_output = gr.Textbox(label="Transcription", placeholder="Transcribed text will appear here...")
            
            # Translation output on the right side
            with gr.Column():
                translation_output = gr.Textbox(label="Translation", placeholder="Translated text will appear here...")
        
        # Translate button
    translate_button = gr.Button("Translate")
        
        # Automatic transcription and detection when audio is uploaded
    audio_input.change(
            fn=handle_transcription_and_detection, 
            inputs=audio_input, 
            outputs=[transcription_output, source_language]  # Update source_language based on detection
        )
        
        # Translation is triggered by the button
    translate_button.click(
            fn=handle_translation, 
            inputs=[transcription_output, target_language], 
            outputs=translation_output
        )

# return interface

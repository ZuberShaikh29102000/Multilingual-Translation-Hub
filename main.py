import gradio as gr
from gradio_toggle import Toggle
from text_to_text import create_interface as tot_interface
from image_to_image import create_interface as img_interface
from ui import create_ui
from speech_to_text import handle_transcription_and_detection, handle_translation

custom_css = """
body {
    position: relative;
    margin: 0;
    padding: 0;
}

#dark-toggle-btn {
    position: fixed;
    top: 10px;
    right: 10px;
    z-index: 999;
}

textarea, input[type="text"] {
    color: black !important;
    background-color: white !important;
}

body.dark {
    background-color: #121212 !important;
    color: white !important;
}

.gradio-toggle-container .toggle-switch {
    border: none !important;
    box-shadow: none !important;
}
"""

with gr.Blocks(theme='allenai/gradio-theme', css=custom_css) as main_app:
    # Place the toggle outside the layout flow so it can be positioned freely
    toggle_dark = Toggle(container=False, value=False, elem_id="dark-toggle-btn")

    toggle_dark.change(
        None,
        inputs=[toggle_dark],
        outputs=[],
        js="""
        (is_dark) => {
            if (is_dark) {
                document.body.classList.add('dark');
            } else {
                document.body.classList.remove('dark');
            }
        }
        """
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("""<h1 style='text-align: center;'>Multilingual Translation Hub</h1>""")

    with gr.Tabs():
        with gr.Tab("Text-to-Text Translation"):
            tot_interface()

        with gr.Tab("Speech-to-Text Translation"):
            create_ui(handle_transcription_and_detection, handle_translation)

        with gr.Tab("Image-to-Image Translation"):
            img_interface()

        
main_app.launch(share=True)

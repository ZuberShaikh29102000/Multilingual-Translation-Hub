import cv2
import fasttext
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import gradio as gr
# import re
# import io
import logging
from paddleocr import PaddleOCR
from language_code2 import language_code_mapping
from fontaddress import nllb_to_font_path
from gradio_toggle import Toggle
import hashlib
import unicodedata

# Set PaddleOCR logging level to INFO to reduce debug output
logging.getLogger('ppocr').setLevel(logging.INFO)

# Build mapping from FastText code to language name
fasttext_code_to_language_name = {}
for language_name, codes in language_code_mapping.items():
    fasttext_code = codes.get('fasttext')
    if fasttext_code:
        fasttext_code_to_language_name[fasttext_code] = language_name

# Load FastText language detection model
fasttext_model = fasttext.load_model('C:/Users/Zuber Shaikh/OneDrive/Desktop/multi/Asset/model/lid.176.bin')  

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1

# Load NLLB-200 translation model with GPU support if available
translator = pipeline("translation", model="facebook/nllb-200-distilled-600M", device=device)

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')  

# Initialize input language with a detection placeholder
INITIAL_LANG_DETECTION = "Detecting Language..."

# Define the available language codes from the mapping
LANGUAGE_CODES = {k: v['fasttext'] for k, v in language_code_mapping.items() if 'fasttext' in v}
LANGUAGE_CHOICES = list(LANGUAGE_CODES.keys())

# Extract text and bounding boxes from OCR
def ocr_text_extraction(image):
    # Perform OCR on the uploaded image
    result = ocr.ocr(np.array(image), cls=True)

    # Extract text and bounding boxes from OCR result
    text_boxes = []
    for line in result:
        for word_info in line:
            text = word_info[1][0]
            box = word_info[0]
            text_boxes.append((text, box))

    return text_boxes

# Detect language using FastText
def detect_language(text):
    clean_text = text.replace("\n", " ").strip()
    if clean_text:
        lang_prediction = fasttext_model.predict(clean_text)
        lang_code = lang_prediction[0][0].replace("__label__", "")
        return lang_code
    return "und"  # undefined

# Translate text using NLLB-200 with language code mapping
def translate_text(text, source_lang_code, target_lang):
    try:
        language_name = fasttext_code_to_language_name.get(source_lang_code)
        if not language_name:
            print(f"Unknown source language code: {source_lang_code}")
            return text

        src_lang_nllb_code = language_code_mapping.get(language_name, {}).get('nllb', source_lang_code)
        tgt_lang_nllb_code = language_code_mapping.get(target_lang, {}).get('nllb', target_lang)

        translated = translator(text, src_lang=src_lang_nllb_code, tgt_lang=tgt_lang_nllb_code)
        translation_text = translated[0]['translation_text']
        print(f"Translated '{text}' from {src_lang_nllb_code} to {tgt_lang_nllb_code}: {translation_text}")
        return translation_text
    except Exception as e:
        print(f"Error translating text '{text}': {e}")
        return text  

# Create a mask for inpainting
def create_inpaint_mask(image_shape, text_boxes):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for _, box in text_boxes:
        x_coords = [point[0] for point in box]
        y_coords = [point[1] for point in box]
        x, y = int(min(x_coords)), int(min(y_coords))
        w, h = int(max(x_coords) - x), int(max(y_coords) - y)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    return mask

# Inpaint the image to remove text
def inpaint_image(image, mask):
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return inpainted_image

# Render translated text on the image after inpainting
def render_translated_text_on_image(image, text_boxes, source_lang_code, target_lang):
    
    # Get the target language NLLB code
    tgt_lang_nllb_code = language_code_mapping.get(target_lang, {}).get('nllb', 'eng_Latn')
    # Select font_path from nllb_to_font_path
    font_path = nllb_to_font_path.get(tgt_lang_nllb_code, "C:/Users/Zuber Shaikh/OneDrive/Desktop/multi/Asset/fonts/NotoSans-Regular.ttf")
    
    mask = create_inpaint_mask(image.shape, text_boxes)
    inpainted_image = inpaint_image(image, mask)

    img_pil = Image.fromarray(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    # font_path = "C:/Users/Zuber Shaikh/Downloads/Noto_Sans/static/NotoSans-Regular.ttf"  # Update this path for your system

    for text, box in text_boxes:
        translated_text = translate_text(text, source_lang_code, target_lang)
        # Normalize the translated text to NFC form
        translated_text = unicodedata.normalize('NFC', translated_text)
        x, y = box[0][0], box[0][1]
        font = ImageFont.truetype(font_path, 16)
        draw.text((x, y), translated_text, fill="black", font=font)

    return img_pil

# Function to draw bounding boxes on the image
def draw_bounding_boxes_on_image(original_image, text_boxes):
    # Ensure we're drawing on a fresh copy of the original image
    img_cv2 = cv2.cvtColor(np.array(original_image.copy()), cv2.COLOR_RGB2BGR)
    for _, box in text_boxes:
        pts = np.array(box, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img_cv2, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    img_with_boxes = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
    return img_with_boxes

# Function to handle image upload
def on_image_upload(image, input_language, user_selected_language, prev_image_hash, processing_active, original_image, cached_boxed_image, toggle_state, updating_image_input):
    # Check if processing is already active
    if processing_active:
        print("Processing is already active; skipping re-processing.")
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), original_image, cached_boxed_image, updating_image_input, gr.update(), gr.update(), gr.update()

    # Check if we're updating the image_input ourselves
    if updating_image_input:
        print("Image input is being updated by the app; skipping processing.")
        updating_image_input = False
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), original_image, cached_boxed_image, updating_image_input, gr.update(), gr.update(), gr.update()

    # Set processing active to prevent re-triggering
    processing_active = True

    if image is None:
        user_selected_language = False
        processing_active = False
        cached_boxed_image = None
        return (
            gr.update(choices=[INITIAL_LANG_DETECTION] + LANGUAGE_CHOICES, value=INITIAL_LANG_DETECTION),
            user_selected_language,
            None,  # Clear previous image hash
            None,  # Clear text_boxes_state
            gr.update(value=False),  # Reset processing flag
            original_image,  # No change to original_image
            cached_boxed_image,
            updating_image_input,
            gr.update(),  # For image_input
            gr.update(visible=False),  # For text_input
            gr.update(visible=False)   # For submit_button
        )

    # Compute hash of the uploaded image
    image_bytes = image.tobytes()
    image_hash = hashlib.md5(image_bytes).hexdigest()

    # Skip processing if the hash matches the previous image
    if image_hash == prev_image_hash:
        print("Image hash matches previous upload; skipping processing.")
        processing_active = False
        return (
            gr.update(),  # No change to input_language
            user_selected_language,
            prev_image_hash,
            gr.update(),  # No change to text_boxes_state
            gr.update(value=False),  # Reset processing flag
            original_image,  # No change to original_image
            cached_boxed_image,
            updating_image_input,
            gr.update(),  # For image_input
            gr.update(),  # For text_input
            gr.update()   # For submit_button
        )

    print("New image detected; processing...")
    text_boxes = ocr_text_extraction(image)
    concatenated_text = "\n".join([text for text, _ in text_boxes])
    source_lang_code = detect_language(concatenated_text)
    detected_lang = fasttext_code_to_language_name.get(source_lang_code, "Unknown")

    # Update the input_language dropdown with the detected language
    dropdown_update = gr.update(
        choices=[f"Detected: {detected_lang}"] + [INITIAL_LANG_DETECTION] + LANGUAGE_CHOICES,
        value=f"Detected: {detected_lang}"
    )

    # Update original_image_state with the newly uploaded image
    original_image = image.copy()

    # Reset cached_boxed_image
    cached_boxed_image = None

    # Reset processing_active to False after processing is complete
    processing_active = False

    # Depending on toggle state, update image_input
    if toggle_state:
        # Draw bounding boxes on a fresh copy of the original image
        updated_image = draw_bounding_boxes_on_image(original_image, text_boxes)
        cached_boxed_image = updated_image  # Cache it
        # Set updating_image_input to True before updating image_input
        updating_image_input = True
        image_input_update = gr.update(value=updated_image)
        # Update text_input with extracted text and make it visible
        text_input_update = gr.update(value=concatenated_text, visible=True)
        submit_button_update = gr.update(visible=True)
    else:
        updated_image = original_image
        # Set updating_image_input to True before updating image_input
        updating_image_input = True
        image_input_update = gr.update(value=original_image)
        text_input_update = gr.update(visible=False)
        submit_button_update = gr.update(visible=False)

    return (
        dropdown_update,
        user_selected_language,
        image_hash,
        text_boxes,
        gr.update(value=False),  # Reset processing flag
        original_image,
        cached_boxed_image,
        updating_image_input,
        image_input_update,
        text_input_update,
        submit_button_update
    )

# Update image with bounding boxes based on toggle state
def update_image_wrapper(toggle_state, original_image, text_boxes, cached_boxed_image, updating_image_input):
    if original_image is None:
        return gr.update(), cached_boxed_image, updating_image_input, gr.update(), gr.update()  # Nothing to update

    if toggle_state and text_boxes:  # Only draw boxes if toggle is enabled and text_boxes are available
        if cached_boxed_image is None:
            print("Toggle enabled: Drawing bounding boxes.")
            # Draw bounding boxes on a fresh copy of the original image
            img_with_boxes = draw_bounding_boxes_on_image(original_image, text_boxes)
            cached_boxed_image = img_with_boxes  # Cache the boxed image
        else:
            print("Using cached image with bounding boxes.")
        # Set updating_image_input to True before updating image_input
        updating_image_input = True
        image_input_update = gr.update(value=cached_boxed_image)
        # Update text_input with extracted text and make it visible
        concatenated_text = "\n".join([text for text, _ in text_boxes])
        text_input_update = gr.update(value=concatenated_text, visible=True)
        submit_button_update = gr.update(visible=True)
    else:
        print("Toggle disabled: Showing original image.")
        # Set updating_image_input to True before updating image_input
        updating_image_input = True
        image_input_update = gr.update(value=original_image)
        text_input_update = gr.update(visible=False)
        submit_button_update = gr.update(visible=False)

    return image_input_update, cached_boxed_image, updating_image_input, text_input_update, submit_button_update

# Function to handle corrections submitted by the user
def on_submit_corrections(text_input_value, text_boxes):
    # Split the corrected text by lines
    corrected_texts = text_input_value.strip().split('\n')
    # Ensure the number of corrected texts matches the number of detected text boxes
    if len(corrected_texts) != len(text_boxes):
        message = "The number of corrected texts does not match the number of detected text boxes."
        print(message)
        # Return the message along with text_boxes_state unchanged
        return text_boxes, gr.update(value=f"<p style='color:red;'>{message}</p>")
    else:
        # Update the texts in text_boxes with the corrected texts
        for i in range(len(text_boxes)):
            text_boxes[i] = (corrected_texts[i], text_boxes[i][1])
        message = "Corrections submitted successfully."
        # Return the updated text_boxes_state and the message
        return text_boxes, gr.update(value=f"<p style='color:green;'>{message}</p>")

# Translate button click logic
def process_image(input_language, original_image, text_boxes, target_language):
    if original_image is None:
        return None  # No image provided

    if not text_boxes:
        text_boxes = ocr_text_extraction(original_image)  # Extract text if missing

    source_language_code = "und"  # Default to undefined
    if input_language.startswith("Detected: "):
        detected_lang = input_language[len("Detected: "):]
        source_language_code = language_code_mapping.get(detected_lang, {}).get('fasttext', 'und')
    elif input_language != INITIAL_LANG_DETECTION:
        source_language_code = LANGUAGE_CODES.get(input_language, "und")

    img_cv2 = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    translated_image = render_translated_text_on_image(img_cv2, text_boxes, source_language_code, target_language)
    return translated_image

# Combine interfaces with dynamic updating
def create_interface():
    with gr.Blocks() as combined_interface:
        gr.Markdown("""<h2 style='text-align: center;'>Image-to-Image Translation</h2>""")
        gr.Markdown("""<h6 style='text-align: left;'>•Note: It only translates English words from the image</h6>""")
        toggle = Toggle(label="Edit the words", value=False)

        # State variables
        user_selected_language = gr.State(False)
        prev_image_hash = gr.State(None)
        text_boxes_state = gr.State(None)  # To store text_boxes
        processing_active = gr.State(False)  # State to prevent re-processing the same image
        original_image_state = gr.State(None)  # To store the original uploaded image
        cached_boxed_image = gr.State(None)  # To cache the image with bounding boxes
        updating_image_input = gr.State(False)  # To indicate if we're updating image_input ourselves

        with gr.Row():
            with gr.Column():
                # Dropdown for input language (above the image upload)
                input_language = gr.Dropdown(
                    choices=[INITIAL_LANG_DETECTION] + LANGUAGE_CHOICES,
                    label="Input Language",
                    value=INITIAL_LANG_DETECTION
                )

                # Image input component
                image_input = gr.Image(type="pil", label="Uploaded Image")

                # Button to process image
                translate_button = gr.Button("Translate")

            with gr.Column():
                # Dropdown for target language (moved above the output image)
                target_language = gr.Dropdown(
                    choices=list(language_code_mapping.keys()),
                    label="Target Language"
                )

                # Output image (processed translation result without bounding boxes)
                output_image = gr.Image(type="pil", label="Translated Image", format="png")

        # Define text_input and submit_button upfront with initial visibility set to False
        text_input = gr.Textbox(
            label="Extracted Texts",
            placeholder="Extracted texts will appear here...",
            visible=False,
            interactive=True,
            lines=10
        )
        submit_button = gr.Button("Submit Corrections", visible=False, interactive=True)

        # Notification message component
        notification = gr.HTML()

        # When the image is uploaded, detect language and update dropdown, and get text_boxes
        image_input.change(
            fn=on_image_upload,
            inputs=[image_input, input_language, user_selected_language, prev_image_hash,
                    processing_active, original_image_state, cached_boxed_image, toggle, updating_image_input],
            outputs=[input_language, user_selected_language, prev_image_hash, text_boxes_state,
                     processing_active, original_image_state, cached_boxed_image, updating_image_input,
                     image_input, text_input, submit_button]
        )

        # Single toggle change handler to manage image update and component visibility
        toggle.change(
            fn=update_image_wrapper,
            inputs=[toggle, original_image_state, text_boxes_state, cached_boxed_image, updating_image_input],
            outputs=[image_input, cached_boxed_image, updating_image_input, text_input, submit_button]
        )

        # When the submit button is clicked, update the text_boxes_state with corrected texts and show a notification
        submit_button.click(
            fn=on_submit_corrections,
            inputs=[text_input, text_boxes_state],
            outputs=[text_boxes_state, notification]
        )

        # When the translate button is clicked, process the image without bounding boxes
        translate_button.click(
            fn=process_image,
            inputs=[input_language, original_image_state, text_boxes_state, target_language],
            outputs=output_image
        )

        # When the user manually selects a language, set user_selected_language to True
        def on_input_language_change(selection):
            return True

        input_language.change(
            fn=on_input_language_change,
            inputs=input_language,
            outputs=user_selected_language
        )

        # Layout for the toggle components
        with gr.Column():
            toggle
            text_input
            submit_button
            notification

    return combined_interface




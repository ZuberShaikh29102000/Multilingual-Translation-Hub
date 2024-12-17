from language_code2 import language_code_mapping
script_to_font = {
    'Latn': 'NotoSans-Regular.ttf',
    'Grek': 'NotoSans-Regular.ttf',
    'Cyrl': 'NotoSans-Regular.ttf',
    'Arab': 'NotoSansArabic-Regular.ttf',
    'Deva': 'NotoSansDevanagari-Regular.ttf',
    'Beng': 'NotoSansBengali-Regular.ttf',
    'Hans': 'NotoSansSC-Regular.ttf',
    'Jpan': 'NotoSansJP-Regular.ttf',
    'Hang': 'NotoSansKR-Regular.ttf',
    'Taml': 'NotoSansTamil-Regular.ttf',
    'Thai': 'NotoSansThai-Regular.ttf',
    'Tfng': 'NotoSansTifinagh-Regular.ttf',
}


base_path = "C:/Users/Zuber Shaikh/OneDrive/Desktop/multi/Asset/fonts/"

nllb_to_font_path = {}

for lang, codes in language_code_mapping.items():
    nllb_code = codes['nllb']
    script_code = nllb_code.split('_')[-1]
    font_file = script_to_font.get(script_code, 'NotoSans-Regular.ttf')
    full_path = base_path + font_file
    nllb_to_font_path[nllb_code] = full_path



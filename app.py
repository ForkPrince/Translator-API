from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from flask import Flask, request, jsonify
from waitress import serve

app = Flask(__name__)

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

supported = ["ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX", "et_EE", "fi_FI", "fr_XX", "gu_IN", "hi_IN", "it_IT", "ja_XX", "kk_KZ", "ko_KR", "lt_LT", "lv_LV", "my_MM", "ne_NP", "nl_XX", "ro_RO", "ru_RU", "si_LK", "tr_TR", "vi_VN", "zh_CN", "af_ZA", "az_AZ", "bn_IN", "fa_IR", "he_IL", "hr_HR", "id_ID", "ka_GE", "km_KH", "mk_MK", "ml_IN", "mn_MN", "mr_IN", "pl_PL", "ps_AF", "pt_XX", "sv_SE", "sw_KE", "ta_IN", "te_IN", "th_TH", "tl_XX", "uk_UA", "ur_PK", "xh_ZA", "gl_ES", "sl_SI"]

@app.route("/", methods=["GET", "POST"])
def translate():
    if request.method == "POST":
        data = request.json

        text = data.get("text")
        source = data.get("source")
        target = data.get("target")

        if source not in supported:
            return jsonify({ "error": "Unsupported source language." }), 400

        if target not in supported:
            return jsonify({ "error": "Unsupported target language." }), 400

        tokenizer.src_lang = source
        encoded = tokenizer(text, return_tensors="pt")

        tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.lang_code_to_id[target]
        )

        translation = tokenizer.batch_decode(tokens, skip_special_tokens=True)

        if len(translation) > 0:
            return jsonify({ "translation": translation[0] })
        else:
            return jsonify({ "error": "Failed to generate output. Output Text Array is empty." }), 500

    elif request.method == "GET":
        return jsonify({ "languages": supported })

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=5000)

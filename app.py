import gradio as gr
from transformers import BertTokenizerFast
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from translation_pipeline import TranslationPipeline
from translator import Translator

model_id = "indiejoseph/bart-translation-zh-yue-onnx"
tokenizer = BertTokenizerFast.from_pretrained(model_id)
model = ORTModelForSeq2SeqLM.from_pretrained(model_id, use_cache=False)

pipe = TranslationPipeline(model=model, tokenizer=tokenizer)
translator = Translator(pipe, batch_size=2, max_length=1024)


def translate(zh: str):
    return translator([zh])[0]


demo = gr.Interface(
    fn=translate,
    inputs=[
        gr.Textbox(label="官話", type="text"),
    ],
    outputs=[
        gr.Textbox(label="廣東話", type="text"),
    ],
    examples=[["瞧瞧你说的是人话吗？"], ["余文乐关掉潮店被嘲临走还坑人"]],
)

if __name__ == "__main__":
    demo.launch(show_api=False)

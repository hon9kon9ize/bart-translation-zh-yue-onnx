"""
ModelHandler defines an example model handler for load and inference requests for MXNet CPU models
"""
from typing import Dict, List, Any
from transformers import AutoTokenizer

from optimum.onnxruntime import ORTModelForSeq2SeqLM
from translator import Translator
from translation_pipeline import TranslationPipeline


class EndpointHandler:
    def __init__(self, path=""):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = ORTModelForSeq2SeqLM.from_pretrained(
            path,
            provider="CPUExecutionProvider",
            encoder_file_name="encoder_model_quantized.onnx",
            decoder_file_name="decoder_model_quantized.onnx",
            decoder_file_with_past_name="decoder_with_past_model_quantized.onnx",
        )
        self.pipe = TranslationPipeline(self.model, self.tokenizer)
        self.translator = Translator(self.pipe, max_length=512, batch_size=1)

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        inputs = data.pop("inputs", data)
        outputs = self.translator(inputs)[0]

        return outputs


# if __name__ == "__main__":
#     my_handler = EndpointHandler(path=".")

#     output = my_handler({"inputs": "今天天氣很好"})

#     print(output)

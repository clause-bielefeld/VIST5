from abc import ABC, abstractmethod
from typing import List
from transformers import T5Tokenizer, T5ForConditionalGeneration
from web_api.utils.onnx_models import OnnxT5
from web_api.utils.ort_settings import get_onnx_runtime_sessions

class LLM(ABC):
    @abstractmethod
    def load_model_from_path(self):
        pass
    @abstractmethod
    def generate(self):
        pass

class LLMImpl(LLM):
    def __init__(self):
        self.thinking_mode = "dialog" # ["inner_monolog", "dialog"]
        self.model = None
    
    def load_model_from_path(self, model_path: str, tokenizer_path: str, model_max_length: str = 2048, is_onnx_model: bool = False):
        if(is_onnx_model):
            self.is_onnx_model = is_onnx_model
            print("Loading ONNX model and tokenizer from path: ", model_path, " ...")
            onnx_encoder_path = model_path + "/vist5_base_model-encoder.onnx"
            onnx_decoder_path = model_path + "/vist5_base_model-decoder.onnx"
            onnx_initial_decoder_path = model_path + "/vist5_base_model-init-decoder.onnx"
            onnx_model_paths = [onnx_encoder_path, onnx_decoder_path, onnx_initial_decoder_path]
            model_sessions = get_onnx_runtime_sessions(onnx_model_paths)
            self.model = OnnxT5(model_path, model_sessions)
            tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, model_max_length=model_max_length) #  tokenizer none type error: solution found here: https://stackoverflow.com/questions/65854722/huggingface-albert-tokenizer-nonetype-error-with-colab
            self.tokenizer = tokenizer
        else:
            self.is_onnx_model = is_onnx_model
            # load model from path -> either local path or huggingface model hub
            print("Loading model and tokenizer from path: ", model_path, " ...")
            model = T5ForConditionalGeneration.from_pretrained(model_path) # './web_api/models/flan_t5_base'
            tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, model_max_length=model_max_length) #  tokenizer none type error: solution found here: https://stackoverflow.com/questions/65854722/huggingface-albert-tokenizer-nonetype-error-with-colab
            self.tokenizer = tokenizer
            self.model = model
        print("... model and tokenizer loaded successfully.")

    def pre_train_model(self, train_data: List[str]):
        # pre-train the model on the provided data
        # next token prediction paradigm?
        pass

    def fine_tune_model(self, train_data: List[str]):
        '''
        INPUT: Fine Tune das Language Model!
        OUTPUT: finet_tune_model(data)
        '''
        # fine-tune the model on the provided data
        # 1. generic react training data/ react instruction fine tuning -> see data from paper: https://arxiv.org/pdf/2210.03629.pdf, 
        # 2. task training data
        # 
        # general fine tuning ideas: 
        # - react fine tuning (=inner monologue types)
        # - instruction fine tuning (=task/input types), chain of thought prompting (=response types)
        # - dialog act fine tuning (=dialog act types)
        # - REASONING fine tuning? generic pre-training? generic reasoning pre-training? 
        pass

    def online_learning(self, train_data: List[str]):
        # online learning paradigm
        pass

    def generate(self, input_string: str):
        if(self.is_onnx_model):
            tokenenized_input = self.tokenizer(input_string, return_tensors="pt")
            input_ids = tokenenized_input["input_ids"]
            attention_mask = tokenenized_input["attention_mask"] 
            output_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=1, min_length=2, max_new_tokens=256, temperature=0.3) #temperature=0.3 
            output_string = self.tokenizer.decode(output_ids.squeeze(), skip_special_tokens=True)
        else:
            # generate text from input string
            tokenized_input = self.tokenizer(input_string, return_tensors="pt") 
            input_ids = tokenized_input.input_ids
            attention_mask = tokenized_input.attention_mask
            outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=1, max_new_tokens=256, min_length=2, temperature=0.3) # temperature=0.5, top_k=50, top_p=0.95, repetition_penalty=1.5, do_sample=True,
            output_string = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output_string
         


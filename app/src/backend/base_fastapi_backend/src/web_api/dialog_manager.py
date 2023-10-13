from abc import ABC, abstractmethod
from typing import List
from transformers import T5Tokenizer, T5ForConditionalGeneration

class DialogManager(ABC):
    @abstractmethod
    def classify_intend(self, input_string: str, context: List[str]):
        pass

    def generate_response(self, input_string: str, context: List[str]):
        pass

class DialogManagerImpl(DialogManager):
    def __init__(self):
        self.dialog_state = {} # slot-value pairs, -> the dialog state contains the current TABLE state and the VISUALIZATION SPEC state
        self.dialog_history = []
        self.memory_management_method = "full_buffer" # [full_buffer, buffer_window, full_summary, summary_window, setfit_memory]
        self.intent_catalog = []
        self.dialog_mode = "open_dialog" # [open_dialog, restricted_dialog, structured_flow_dialog, rlhf_dialog]
        # alternative modes based on long term memory: [no_augmentation, knowledge_base_augmentation, reasoning?, ....]
        self.context = [] # context for the response generation
        # load the model
        self.load_model()
        print(str(self.__class__.__name__), " initialized successfully.")

    def classify_intend(self, input_string: str, context: List[str]):
        pass

    def generate_response(self, input_string: str):
        # check the mode
        if self.dialog_mode == "open_dialog":
            # generate response
            response = self.model_generate(input_string)
            return response
        elif self.dialog_mode == "restricted_dialog":
            # classify intend 
            # populate context
            # generate response
            pass
        elif self.dialog_mode == "structured_flow_dialog":
            # generate response
            pass
        elif self.dialog_mode == "rlhf_dialog":
            # generate response
            pass
        else:
            # raise error
            raise ValueError("ERROR: Invalid dialog mode provided.")

    def load_model(self):
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.tokenizer = tokenizer
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.model = model
        print("Model loaded successfully.")
        #input_ids = tokenizer("translate English to German: The house is extremely clean and nice to look at.", return_tensors="pt").input_ids
        #outputs = model.generate(input_ids)
        #print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    def model_generate(self, input_string: str):
        input_ids = self.tokenizer(input_string, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids)
        output_string = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output_string

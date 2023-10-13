from abc import ABC, abstractmethod
from typing import List
import json 
import ast 

class ShortTermMemory(ABC):
    pass

class ShortTermMemoryImpl(ShortTermMemory):
    def __init__(self, short_term_memory_mode: str = "buffer_window", buffer_window_size: int = 6):
        # set memory mode
        self.short_term_memory_mode = "buffer_window" # ["full_buffer", "buffer_window", "full_summary", "window_summary"]
        self.buffer_window_size = buffer_window_size 
        self.max_sequential_memory_length = 8
        # STATE: the short term memory contains the STATE of the dialog agent. this state is a partial of the world/environment state. 
        # initialize sequential memory
        self.sequential_memory = [] # = dialog/chat history 
        # initialize graph memory = contains objects, concepts, relations
        self.graph_memory = {'table_state': '', 'visualization_state': {'mark':'none', 'color':'none'} } # = STATE in our case {"table_state":"", "visualization_state":{}, "agent_state":"", "dialog_state":""}

    def add_utterance_to_sequential_memory(self, utterance: str):
        # add utterance to sequential memory
        self.sequential_memory.append(utterance)

    # SOPHISTICATED VERSION: important: python does not support overload of functions -> instead we use default values for the parameters
    def update_graph_memory(self, memory_object: str, memory_value: str, memory_key: str = None):
        # check if memory_key is None
        if memory_key is None:
            # update graph memory object with the given value
            self.graph_memory[memory_object] = memory_value
        else:
            # update graph memory object at the given key with the given value
            self.graph_memory[memory_object][memory_key] = memory_value
        
    def reset_graph_memory_object(self, memory_object: str): 
        # reset graph memory object to empty dictionary
        self.graph_memory[memory_object] = {} 

    def add_to_chat_history(self, input_string: str, output_string: str):
        # add input string to sequential memory, IMPORTANT: INPUT is automatically added by the PROMPT TEMPLATE
        input_string =  input_string
        self.sequential_memory.append(input_string)
        # add response to sequential memory
        output_string = "OUTPUT: " + output_string
        self.sequential_memory.append(output_string)
        # MEMORY SIZE CHECK: check if sequential memory is too long
        if len(self.sequential_memory) > self.max_sequential_memory_length:
            # remove first element
            self.sequential_memory.pop(0)

    def get_chat_history(self):  
        if(self.short_term_memory_mode == "full_buffer"):
            # format chat history to string, add line break after each utterance
            chat_history = ""
            for index, utterance in enumerate(self.sequential_memory): 
                # add line break after each utterance if not last utterance
                if index < len(self.sequential_memory)-1:
                    chat_history += utterance + "\n"
                else:
                    chat_history += utterance
        elif(self.short_term_memory_mode == "buffer_window"):
            # format chat history to string, add line break after each utterance
            chat_history = ""
            for index, utterance in enumerate(self.sequential_memory[-self.buffer_window_size:]): 
                # add line break after each utterance if not last utterance
                if index < len(self.sequential_memory[-self.buffer_window_size:])-1:
                    chat_history += utterance + "\n"
                else:
                    chat_history += utterance
        # check if chat_history is longer than 9000 characters, if so, truncate it
        if len(chat_history) > 8000:
            chat_history = chat_history[-8000:]
        return chat_history

    def visualization_state_to_string(self):
        # get the visualization state of the agent which is stored in the graph memory
        visualization_state_dict = self.graph_memory['visualization_state']
        # make sure that the list in the ['data']['values'] key, IF IT EXISTS, is not longer than 5 elements, because this blows up the context of the LLM!
        if 'data' in visualization_state_dict:
            if 'values' in visualization_state_dict['data']:
                if len(visualization_state_dict['data']['values']) > 5:
                    visualization_state_dict['data']['values'] = visualization_state_dict['data']['values'][:5]
        # convert dict to string
        visualization_state = self.flatten_dict_and_stringify(visualization_state_dict)
        return visualization_state

    def get_current_state(self):
        # get current state of the agent which is stored in the graph memory
        current_table_state = "Table State:\n" + self.graph_memory["table_state"]
        visualization_state_string = self.visualization_state_to_string()
        current_visualization_state = "Visualization State:\n" + visualization_state_string
        current_state = ""
        current_state += current_table_state + "\n" #"\n##\n"
        current_state += current_visualization_state
        return current_state

    def flatten_dict_and_stringify(self, d, separator='_', list_separator='-'):
        final = {}
        def _flatten_dict(obj, parent_keys=[]):
            for k, v in obj.items(): 
                if isinstance(v, dict): 
                    _flatten_dict(v, parent_keys + [k])
                elif isinstance(v, list):
                    for i, item in enumerate(v):
                        if isinstance(item, dict):
                            _flatten_dict(item, parent_keys + [k + list_separator + str(i)])
                        else:
                            key = separator.join(parent_keys + [k + list_separator + str(i)])
                            final[key] = item
                else:
                    key = separator.join(parent_keys + [k])
                    final[key] = v
        
        _flatten_dict(d)
        # stringify
        final_str = json.dumps(final) # final_str = str(final)
        # remove brackets {} because they can not be tokenized
        final_str = final_str[1:-1]
        return final_str
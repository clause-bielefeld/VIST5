from abc import ABC, abstractmethod
from typing import List
from web_api.llm import LLMImpl
from web_api.memory import MemoryImpl
from web_api.action_space import ActionSpaceImpl
import pandas as pd
import os
import re
import ast 
import json 
import collections.abc
import random
import string

class Agent(ABC):
    @abstractmethod
    def generate_output(self, input_string: str, context: List[str]):
        pass
    @abstractmethod
    def text_to_speech(self, input_string: str):
        pass
    @abstractmethod
    def speech_to_text(self, input_string: str):
        pass

class AgentImpl(Agent):
    def __init__(self, llm_path: str, llm_tokenizer_path:str, similarity_model_path: str, fine_tune_similarity_model: bool, document_db_file_path: str, vector_store_file_path: str, agent_type: str = "conversational_agent", dialog_mode: str = "retrieval_augmented_dialog", max_doc_length: int = 5000):
        self.agent_type = agent_type
        self.llm_path = llm_path
        self.dialog_mode = dialog_mode # [open_dialog, retrieval_augmented_dialog, structured_flow_dialog, rlhf_dialog]
        self.max_doc_length = max_doc_length # on avg 1000 tokens have 4700 characters in the english language -> 5000 characters is a good max length
        # ONLINE LEARNING
        self.online_training_samples_db_json_file_path = "./web_api/data/online_training_samples_db.json"
        if os.path.isfile(self.online_training_samples_db_json_file_path):
            # load online_training_samples_db_json_file_path
            print('... loading online_training_samples_db_json_file_path')
            with open(self.online_training_samples_db_json_file_path, 'r') as f:
                self.online_training_samples_db = json.load(f)
        else: 
            # generate online_training_samples_db json file
            print('... generating online_training_samples_db_json_file_path')
            self.online_training_samples_db = []
            # save document db to json file 
            with open(self.online_training_samples_db_json_file_path, 'w') as f:
                json.dump(self.online_training_samples_db, f)
        # initialize the large language model class
        self.model = LLMImpl()
        # load the model from path
        self.model.load_model_from_path(model_path=llm_path, tokenizer_path=llm_tokenizer_path, is_onnx_model=True)
        # initialize the memory -> contains short term memory and long term memory
        self.memory = MemoryImpl(similarity_model_path=similarity_model_path, fine_tune_similarity_model=fine_tune_similarity_model, document_db_file_path=document_db_file_path, vector_store_file_path=vector_store_file_path)
        # ACTION SPACE: initialize the action space of the agent
        self.action_space = ActionSpaceImpl(agent=self)
        # INITALIZE AGENT STATE
        # set the graph memory/state of the agent = table state and visualization state, set the initial dataframe (= is updated when the update_table_state function is called)
        self.current_dataframe = pd.DataFrame() 
        self.current_result_dataframe = pd.DataFrame() 
        self.current_table_name = "None" # initial value -> will be set to weather in update_table_state function below
        initial_table_file_name = "northern_european_cities.csv" 
        self.update_table_state(new_table_file_name=initial_table_file_name)
        initial_visualization_state = {} # TODO: how do we want to define the initial visualization state?
        self.update_visualization_state(new_visualization_state=initial_visualization_state)
        # initalize the sequential memory/dialog history/chat history of the agent
        initial_agent_utterance = "OUTPUT: action: text_response; args: \"text\": \"Hello, I am VIST5. As a specialized visualization chatbot, I can generate visualizations from natural language descriptions, allow modifications based on your requirements, and answer any questions you may have related to your visualization.\";"
        self.memory.short_term_memory.add_utterance_to_sequential_memory(utterance=initial_agent_utterance)
        print(str(self.__class__.__name__), " initialized successfully.")

    def generate_output(self, input_string: str):
        # check the mode
        if self.dialog_mode == "open_dialog":
            # create input string from initial prompt
            open_dialog_prompt_template = self.memory.working_memory.open_dialog_prompt_template
            personality_prompt_template = self.memory.working_memory.personality_prompt_template
            # get chat history from short term memory
            chat_history = self.memory.short_term_memory.get_chat_history()
            user_input = input_string
            #print("chat_history: ", chat_history)
            # assemble input string 
            input_string = open_dialog_prompt_template.format(chat_history=chat_history, user_input=user_input, personality_prompt=personality_prompt_template)
            # generate response
            text_output = self.model.generate(input_string)
            # add input and response to short term memory
            self.memory.short_term_memory.add_to_chat_history(user_input=user_input, response=text_output)
            response = {"action": text_output, "args": []}
            return response 
        elif self.dialog_mode == "retrieval_augmented_dialog":
            user_input = input_string
            # RETRIEVAL: first get a list of document ids and their embeddings from the knowledge base of documents that could help, documents can either be: KNOWLEDGE_NODES, FEW-SHOT-EXAMPLES, or TOOLS
            ## FEW SHOT RETRIEVAL:
            top_k_examples = self.memory.retrieve_examples_from_knowledge_base(query=input_string, k=3) # returns a list of objects with the following structure: [{"input": "example_input", "output": "example_output"}, ...]
            examples = []
            if len(top_k_examples) > 0:
                # create an instruction prompt for the model to make use of the few shot examples 
                instruction_prompt = "Respond to the following sentence using the examples below as a guide. \n\n"
                user_input = instruction_prompt + user_input
                for example in top_k_examples:
                    current_example = "INPUT: " + example['input'] + "\n" + "OUTPUT: " + example['output'] + "\n###\n"
                    examples.append(current_example)
            # check if examples is non-empty
            examples_str = "\n\nHere are some examples:\n{}\n".format("\n".join(examples)) if examples else "" 
            # shorten examples if it is too long 
            if len(examples_str) > self.max_doc_length:
                examples_str = examples_str[:self.max_doc_length]
            ## KNOWLEDGE NODE RETRIEVAL:
            top_k_documents = self.memory.retrieve_documents_from_knowledge_base(query=input_string, k=3) # returns a list of objects with the following structure: [{"input": "example_input", "output": "example_output"}, ...]
            # check if len of top_k_documents is >0
            combined_document_str = ""    
            if len(top_k_documents) > 0: 
                for index, document in enumerate(top_k_documents):
                    # if document is not the last document, then add a new line
                        combined_document_str += document['input'] 
                # shorten context if it is too long
                if len(combined_document_str) > self.max_doc_length:
                    combined_document_str = combined_document_str[:self.max_doc_length]
            # check if context is non-empty
            context_str = "Context: {}\n".format(combined_document_str) if combined_document_str else ""
            # assemble user input
            user_input_prompt = self.memory.working_memory.user_input_prompt_template.format(user_input=user_input, examples=examples_str, context=context_str)
            # get chat history from short term memory
            chat_history = self.memory.short_term_memory.get_chat_history()
            # generate the state prompt from the current state of the agent in the short term memory
            current_state = self.memory.get_current_state()
            state_prompt = current_state 
            # assemble base prompt
            base_prompt =  self.memory.working_memory.base_prompt_template.format(state_prompt=state_prompt, chat_history=chat_history, input_prompt=user_input_prompt)
            #print("base_prompt: ", base_prompt)
            # generate response 
            output_string = self.model.generate(base_prompt)
            # SANITY CHECK: 
            print("base_prompt: ", base_prompt)
            print("output_string: ", output_string)
            action_name, args_list = self.check_model_output(output_string)
            # STATE UPDATE: 
            # add input and response to short term memory
            self.memory.short_term_memory.add_to_chat_history(input_string=user_input_prompt, output_string=output_string) 
            # INTERNAL ACTIONS: check if the model wants to run an internal loop  
            while action_name in self.action_space.internal_loop_actions: # TODO: replace this with a while loop for enabling longer internal loops
                # if yes, then the args_list[0] is the tool_output or internal thinking process output -> tool has already been executed -> now wee need to execute the model again on the tool or internal output
                # get the tool output 
                tool_output = args_list[0]
                # assemble tool input: the tool output is the new context and we need to generate a new response
                tool_input_prompt = self.memory.working_memory.tool_input_prompt_template.format(tool_output=tool_output)
                # get chat history from short term memory
                chat_history = self.memory.short_term_memory.get_chat_history()
                # generate the state prompt from the current state of the agent in the short term memory
                current_state = self.memory.get_current_state()
                state_prompt = current_state 
                # assemble prompt
                base_prompt =  self.memory.working_memory.base_prompt_template.format(state_prompt=state_prompt, chat_history=chat_history, input_prompt=tool_input_prompt)
                # generate response
                output_string = self.model.generate(base_prompt)
                action_name, args_list = self.check_model_output(output_string)
                # SANITY CHECK: 
                print("base_prompt: ", base_prompt)
                print("output_string: ", output_string)
                # STATE UPDATE:  
                # add input and response to short term memory
                self.memory.short_term_memory.add_to_chat_history(input_string=tool_input_prompt, output_string=output_string) 
            # if the action is not an internal loop action, then we have a direct response action and we can return the response
            # ASSEMBLE RESPONSE: 
            response = {"action": action_name, "args": args_list} 
            return response
            # 
        elif self.dialog_mode == "structured_flow_dialog": 
            # generate response
            # get chat history from short term memory
            chat_history = self.memory.short_term_memory.get_chat_history()
            # replace 'OBSERVATION:' in chat history with 'USER:'
            chat_history = chat_history.replace("OBSERVATION:", "USER:")
            # replace 'ACTION:' in chat history with 'VIST5:'
            chat_history = chat_history.replace("ACTION:", "VIST5:")
            user_input = input_string
            # assemble context 
            # RETRIEVAL: first get a list of document ids and their embeddings from the knowledge base of documents that could help, documents can either be: KNOWLEDGE_NODES, FEW-SHOT-EXAMPLES, or TOOLS
            top_k_examples = self.memory.retrieve_examples_from_knowledge_base(query=input_string) # returns a list of objects with the following structure: [{"input": "example_input", "output": "example_output"}, ...]
            context = ""    
            context += "The following are some examples from previous conversations:\n"
            for index, example in enumerate(top_k_examples):
                # if example is not the last example, then add a new line
                if index != len(top_k_examples) - 1:
                    context += "USER: " + example['input'] + "\n" + "VIST5: " + example['output'] + "\n" #"\n--\n"
                else:
                    context += "USER: " + example['input'] + "\n" + "VIST5: " + example['output']
            # shorten context if it is too long
            if len(context) > self.max_doc_length:
                context = context[:self.max_doc_length]
            # assemble prompt
            personality_prompt_template = self.memory.working_memory.personality_prompt_template
            context_prompt_template = self.memory.working_memory.context_prompt_template
            # generate the state prompt from the current state of the agent in the short term memory
            current_state = self.memory.get_current_state()
            state_prompt_template = current_state 
            context_prompt = context_prompt_template.format(chat_history=chat_history, user_input=user_input, context=context, personality_prompt=personality_prompt_template, state_prompt=state_prompt_template)
            #print("context_prompt: ", context_prompt)
            # generate response 
            output_string = self.model.generate(context_prompt)
            action_name, args_list = self.check_model_output(output_string)
            print("context_prompt: ", context_prompt)
            print("output_string: ", output_string)
            # STATE UPDATE: 
            # add input and response to short term memory
            self.memory.short_term_memory.add_to_chat_history(user_input=user_input, response=output_string) #response=text_output)
            # ASSEMBLE RESPONSE: 
            response = {"action": action_name, "args": args_list} 
            return response
        elif self.dialog_mode == "rlhf_dialog":
            # generate response
            pass
        else:
            # raise error
            raise ValueError("ERROR: Invalid dialog mode provided.")
    
    def rerank_documents(self, document_ids: List[str], document_embeddings: List[str], query: str):
        # rerank the documents based on their embeddings and the current state of the agent (= query + context)
        # TODO: 
        return document_ids

    def run_tool(self, action_name:str, args_list:List[str]):
        # get the action_name and then run the corresponding tool function
        if action_name == "search_tool":
            # run search tool with the args_list
            print('running search_tool')
            tool_output = "search_tool_output"
            return tool_output
        elif action_name == "calculator_tool":
            # run calculator tool with the args_list
            print('running calculator_tool')
            tool_output = "calculator_tool_output"
            return tool_output
        elif action_name == "sql_tool":
            # run sql tool with the args_list
            print('running sql_tool')
            tool_output = "sql_tool_output"
            return tool_output
        else:
            # raise error
            raise ValueError("ERROR: Invalid action name provided.")
        return tool_output

    def check_model_output(self, output_string: str):
        # get action name and args_list, model output is structured output like this: action: action_name; args: text=here is some text | arg1=value | argn=value;
        try: 
            # first check if action and args is in string -> otherwise we have generatl NIv2 sample
            if "action:" not in output_string and "args:" not in output_string:
                # then we have a general niv2 example and we want to just give back a text response and the outputstring
                action_name = "text_response"
                args_list = [output_string]
                return action_name, args_list
            else:
                # get action name and args_list by splitting the string at the first semicolon
                output_list = output_string.split(";", 1)
                action_name = output_list[0]
                args_flattened_str = output_list[1]  
                # remove the last semicolon from the args_flattened_str
                args_flattened_str = args_flattened_str[:-1]
        except:
            print('ERROR parsing model output_string: ' + str(output_string))
            action_name = "text_response"
            args_list = ["I am sorry, I did not understand your query. Maybe you can rephrase it?"]
            return action_name, args_list
        # remove the 'action: ' substring from the action_name
        action_name = action_name.replace("action: ", "").strip()
        # remove the 'args: ' substring of the args_list 
        args_flattened_str = args_flattened_str.replace("args: ","")
        # check if action_name is in self.action_list
        print(action_name)
        print(args_flattened_str)
        # check if action_name is in self.action_list
        if action_name in self.action_space.action_list:
            # check if action_name is a direct response action
            if(action_name in self.action_space.direct_response_actions):
                # check if corresponding function to action_name exists in action_space
                if hasattr(self.action_space, action_name):
                    # get the function from the action_space
                    action_function = getattr(self.action_space, action_name)
                    # try to run the function with the inputs
                    try:
                        print('calling function: ', action_name, ' with args: ', args_flattened_str)
                        action_name, frontend_args_list = action_function(args_flattened_str)
                        return action_name, frontend_args_list
                    except Exception as e:
                        print("ERROR: direct response action function could not be executed.")
                        print(e)
                        print("args_flattened_str: ", args_flattened_str)
                        return "text_response", ["I am sorry, I did not understand your query. Maybe you can rephrase it?"]
                else:
                    return "text_response", ["I am sorry, I did not understand your query. Maybe you can rephrase it?"]
            # check if action_name is an internal loop action -> then we do not need to return frontend_args to update the frontend after action execution, but rather need the output for the next internal action. 
            elif(action_name in self.action_space.internal_loop_actions):
                # check if corresponding function to action_name exists in action_space
                if hasattr(self.action_space, action_name):
                    # get the function from the action_space
                    action_function = getattr(self.action_space, action_name)
                    # try to run the function with the inputs
                    try:
                        action_name, args_list = action_function(args_flattened_str)
                        return action_name, args_list
                    except Exception as e:
                        print("ERROR: Action internal loop action function could not be executed.")
                        print(e)
                        print("args_flattened_str: ", args_flattened_str)
                        return "text_response", ["I am sorry, I did not understand your query. Maybe you can rephrase it?"]
                else:
                    return "text_response", ["I am sorry, I did not understand your query. Maybe you can rephrase it?"]
        else:
            print("ERROR: provided action name is not in action_list")
            # return error message
            action_name = "text_response"
            args_list = ["I am sorry, I did not understand your query. Maybe you can rephrase it?"]
            return action_name, args_list

    def text_to_speech(self, text: str):
        # convert text to speech
        speech = self.tts(text)
        return speech

    def speech_to_text(self, speech: str):
        # convert speech to text
        text = self.stt(speech)
        return text

    def update_table_state(self, new_table_file_name: str):
        # create table state from column names of the csv file
        TABLE_BASE_PATH = "/src/web_api/data/tables/"
        new_table_path = TABLE_BASE_PATH + new_table_file_name
        # check if new_table_path exists
        if os.path.exists(new_table_path): 
            # read the csv file using pandas
            self.current_dataframe = pd.read_csv(new_table_path)
            # drop old table, if it exists
            drop_old_table_query = f"DROP TABLE IF EXISTS {self.current_table_name}"
            self.action_space.sql_connection.execute(drop_old_table_query) 
            # TABLE NAME: add the table_name in front of the column names: table_name: name, col: col_name | col_name | col_name
            # get table name -> the table name is the file name without the file extension
            table_name = new_table_file_name.split(".")[0]
            self.current_table_name = table_name
            # connect new table to sqlite 
            self.current_dataframe.to_sql(self.current_table_name, self.action_space.sql_connection, index=False)
            # get header of the table
            header = self.current_dataframe.columns
            # convert header into list of strings 
            header_list = header.astype(str).tolist() 
            # get the dtypes of all columns as a list of strings 
            dtypes = self.current_dataframe.dtypes.astype(str).tolist() 
            # after each column name, add the dtype of the column in brackets like this: "column_name (dtype)"
            header = [column_name + " (" + dtype + ")" for column_name, dtype in zip(header, dtypes)]
            # convert header to string which captures the table state
            header_string = "col : " + " | ".join(header)
            # get data types of the columns
            dtypes_list = self.current_dataframe.dtypes.astype(str).tolist()
            # get first 3 rows of the table
            first_3_rows = self.current_dataframe.head(3)
            # convert first 3 rows into list of lists
            rows = first_3_rows.values.tolist()
            # check if a row exists that is None, if so remove the row from the list
            rows = [row for row in rows if row is not None]
            # turn all elements into strings 
            rows = [[str(element) for element in row] for row in rows]
            rows_string = ""
            for i, row in enumerate(rows):
                rows_string += "row_{} : ".format(i) + " | ".join(row) + " "
            #print("dtypes_list: ", dtypes_list)
            # get the current table state = table
            updated_table_state = f"table_name : {table_name} " + header_string + " " + rows_string
            print("updated_table_state: ", updated_table_state)
            # DIALOG RECORDING: 
            # try to get the last dialog record object from the online training samples db if it exists
            try:
                dialog_record_object = self.online_training_samples_db[-1]
                # if it exists, we check if the turns list is empty 
                if(len(dialog_record_object["turns"]) == 0):
                    # if the turns list is empty, we just overwrite the table state in the dialog record object
                    dialog_record_object["table_state"] = updated_table_state
                else:
                    # if the turns list is not empty, we create a new dialog record object
                    # first we create a random dialog id without any third party library
                    dialog_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
                    dialog_table_state = updated_table_state
                    dialog_turns = []
                    # add dialog record object to the online training samples db 
                    dialog_record_object = {"id": dialog_id, "table_state": dialog_table_state, "turns": dialog_turns}
                    self.online_training_samples_db.append(dialog_record_object)
            except:
                # if it doesnt exist, that means that this is the first one we create a new one 
                # => create a new dialog record object for this dialog -> this is used for online learning later
                # first we create a random dialog id without any third party library
                dialog_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
                dialog_table_state = updated_table_state
                dialog_turns = []
                # add dialog record object to the online training samples db 
                dialog_record_object = {"id": dialog_id, "table_state": dialog_table_state, "turns": dialog_turns}
                self.online_training_samples_db.append(dialog_record_object)
            # UPDATE TABLE STATE: update the table state
            self.memory.short_term_memory.update_graph_memory(memory_object="table_state", memory_value=updated_table_state)
            # create table html as html string, limit the number of rows to 1000
            table_html = self.current_dataframe.head(1000).to_html(index=False, classes="table is-striped")
            # create response as dictionary which contains the table state and the data types of the columns
            response = {"header_list": header_list, "data_types": dtypes_list, "table_html": table_html}
        else:
            # raise error
            print("ERROR: Invalid table path provided.")
            response = "ERROR: Invalid table path provided."
        # return response
        return response

    def update_visualization_state(self, new_visualization_state: str):
        # TODO: this can e.g. happen if a user manually changes code in the frontend that affects the visualization
        # update the visualization state
        self.memory.short_term_memory.update_graph_memory(memory_object="visualization_state", memory_value=new_visualization_state)
        # return response
        response = "success"
        return response

    def update_dictionary(self, original_dict, update_dict):
        """
        Updates an original dictionary with the key-value pairs from an update dictionary. If a key from the update 
        dictionary already exists in the original dictionary, the value for that key will be updated or merged recursively.
        Solution found here: https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth, 
        Alternative:using pydantic is also possible -> see stackoverflow above

        :param original_dict: The original dictionary to be updated
        :param update_dict: The dictionary containing the updates to be applied to the original dictionary
        :return: The updated original dictionary
        """
        # iterate over the key-value pairs in the update dictionary
        for key, value in update_dict.items(): 
            # if the value is itself a dictionary, recursively update the corresponding value in the original dictionary
            if isinstance(value, collections.abc.Mapping):
                try:
                    original_dict[key] = self.update_dictionary(original_dict.get(key, {}), value)
                except Exception as e: 
                    # IMPORTANT: we have to really trust the code generator here to give us a correct object ... -> otherwise just run into the exception and do not update at all ...
                    original_dict[key] = {}
                    original_dict[key] = self.update_dictionary(original_dict.get(key, {}), value)
            # otherwise, simply update the value for the corresponding key in the original dictionary
            else:
                original_dict[key] = value
        # return the updated original dictionary
        return original_dict

    # VISUALIZATION CREATION/VIS PIPELINE
    def create_visualization_from_spec(self, vis_spec_string, result_dataframe):
        # check if result dataframe is empty
        if result_dataframe.empty:
            empty_vis = {"data": {"values": []}, "mark": "bar", "encoding": {"x": {"field": "x", "type": "quantitative"}, "y": {"field": "y", "type": "quantitative"}}}
            # return empty visualization
            return empty_vis 
        # Check for duplicate columns -> if column names have the same name -> rename them with index
        if result_dataframe.columns.duplicated().any():
            # Rename duplicate columns
            result_dataframe.columns = [f"{col}_{idx}" if result_dataframe.columns[:idx].tolist().count(col) > 0 else col for idx, col in enumerate(result_dataframe.columns)]
        # check if visualization spec string is empty 
        if vis_spec_string == "":
            # if the visualization spec is empty, we have to use a visualization recommendation approach that gives us the best vega lite visualization for the result dataframe
            # get the number of columns in the result dataframe
            num_columns = len(result_dataframe.columns)
            # get the number of rows in the result dataframe
            num_rows = len(result_dataframe.index)
            # check if the result_dataframe is empty
            if num_columns == 0 or num_rows == 0:
                # if the result dataframe is empty, we return an empty visualization
                # create empty visualization
                empty_vis = {"data": {"values": []}, "mark": "bar", "encoding": {"x": {"field": "x", "type": "quantitative"}, "y": {"field": "y", "type": "quantitative"}}}
                # return empty visualization
                return empty_vis
            # check if the number of columns is 1
            if num_columns == 1:
                # check if the number of rows is 1
                if num_rows == 1:
                    # SINGLE OBJECT SPECIAL CASE: -> we return a text visualization from vega lite showing the value of the single object
                    # get the value of the single object
                    single_object_value = result_dataframe.iloc[0, 0]
                    # create text visualization
                    text_vis = {"data": {"values": [{"text": str(single_object_value)}]}, "mark": "text", "encoding": {"text": {"field": "text", "type": "nominal"}}}
                    # return text visualization
                    return text_vis
                # check if the number of rows is greater than 1, if so, we visualize the data as a horizontal bar chart where the x axis is the column name and the y axis is the value
                if num_rows > 1:
                    # HORIZONTAL BAR CHART SPECIAL CASE: -> we return a horizontal bar chart from vega lite showing the values of the single column
                    # get the column name
                    column_name = result_dataframe.columns[0]
                    # create horizontal bar chart visualization
                    horizontal_bar_chart_vis = {"data": {"values": result_dataframe.to_dict(orient="records")}, "mark": "bar", "encoding": {"y": {"field": column_name, "type": "nominal", "title":""}}}
                    # return horizontal bar chart visualization
                    return horizontal_bar_chart_vis
            # check if the number of columns is 2 
            if num_columns == 2: 
                # check if the number of rows is 1
                if num_rows == 1:
                    # HORIZONTAL BAR fallback
                    # alternative: point
                    # TODO: check this using the dataframe type -> if both int -> then go for point
                    # get the column names
                    column_name_1 = result_dataframe.columns[0]
                    column_name_2 = result_dataframe.columns[1]
                    mark = "bar"
                    # create horizontal bar chart visualization
                    horizontal_bar_chart_vis = {"data": {"values": result_dataframe.to_dict(orient="records")}, "mark": mark, "encoding": {"x": {"field": column_name_1, "type": "quantitative"}, "y": {"field": column_name_2, "type": "nominal"}}}
                    # return horizontal bar chart visualization
                    return horizontal_bar_chart_vis
                # check if the number of rows is greater than 1, if so, we visualize the data as a scatter plot where the x axis is the first column and the y axis is the second column
                if num_rows > 1:
                    # SCATTER PLOT or BAR CHART fallback -> we return a scatter plot from vega lite showing the values of the two columns
                    # get the column names
                    column_name_1 = result_dataframe.columns[0]
                    column_name_2 = result_dataframe.columns[1]
                    # create scatter plot visualization
                    #scatter_plot_vis = {"data": {"values": result_dataframe.to_dict(orient="records")}, "mark": "point", "encoding": {"x": {"field": column_name_1, "type": "quantitative"}, "y": {"field": column_name_2, "type": "quantitative"}}}
                    # additionally create a bar chart visualization
                    bar_chart_vis = {"data": {"values": result_dataframe.to_dict(orient="records")}, "mark": "bar", "encoding": {"x": {"field": column_name_1, "type": "quantitative"}, "y": {"field": column_name_2, "type": "quantitative"}}}
                    return bar_chart_vis
            # check if the number of columns is 3
            if num_columns == 3:
                # BAR PLOT fallback -> we return a bar plot from vega lite showing the first column on x, the second column on y and the third column as color + we add a color legend
                # get the column names
                column_name_1 = result_dataframe.columns[0]
                column_name_2 = result_dataframe.columns[1]
                column_name_3 = result_dataframe.columns[2]
                # create bar plot visualization
                bar_plot_vis = {"data": {"values": result_dataframe.to_dict(orient="records")}, "mark": "bar", "encoding": {"x": {"field": column_name_1, "type": "quantitative"}, "y": {"field": column_name_2, "type": "quantitative"}, "color": {"field": column_name_3, "type": "nominal"}}, "config": {"legend": {"orient": "bottom"}}}
                return bar_plot_vis
            # check if the number of columns is greater than 3, if so, we visualize the data as a scatter plot where the x axis is the first column and the y axis is the second column
            if num_columns > 3:
                # SCATTER PLOT fallback -> we return a scatter plot from vega lite showing the values of the first two columns
                # get the column names
                column_name_1 = result_dataframe.columns[0]
                column_name_2 = result_dataframe.columns[1]
                # create scatter plot visualization
                scatter_plot_vis = {"data": {"values": result_dataframe.to_dict(orient="records")}, "mark": "point", "encoding": {"x": {"field": column_name_1, "type": "quantitative"}, "y": {"field": column_name_2, "type": "quantitative"}}}
                return scatter_plot_vis
        else:
            # check if dataframe is empty
            if result_dataframe.empty:
                # create empty visualization
                empty_vis = {"data": {"values": []}, "mark": "point", "encoding": {"x": {"field": "x", "type": "quantitative"}, "y": {"field": "y", "type": "quantitative"}}}
                return empty_vis
            # if we have a visualization spec string, we can use it to create the visualization
            # turn the visualization spec string into a dictionary using ast.literal_eval, format: {"mark":"bar", "bin":{"axis":"x", "type":"year"}} 
            visualization_spec_dict = self.unstringify_and_unflatten_dict(vis_spec_string)
            # get the mark type from the visualization spec string 
            mark_type = visualization_spec_dict["mark"]
            # the first column in the result dataframe is the x encoding
            x_encoding = result_dataframe.columns[0]
            # the second column in the result dataframe is the y encoding
            y_encoding = result_dataframe.columns[1]
            # if available: the third column in the result dataframe is the color encoding
            #if num_columns == 3:
            #    color_encoding = result_dataframe.columns[2]
            # create the visualization spec dictionary
            raw_vega_lite_specs = {
                'bar': {
                    "mark": "bar",
                    "encoding": {
                        "x": {"field": "x", "type": "nominal"},
                        "y": {"field": "y", "type": "quantitative"}
                    }
                },
                'arc': {
                    "mark": "arc",
                    "encoding": {
                        "color": {"field": "x", "type": "nominal"},
                        "theta": {"field": "y", "type": "quantitative"}
                    }
                },
                'line': {
                    "mark": "line",
                    "encoding": {
                        "x": {"field": "x", "type": "nominal"},
                        "y": {"field": "y", "type": "quantitative"}
                    }
                },
                'point': {
                    "mark": "point",
                    "encoding": {
                        "x": {"field": "x", "type": "quantitative"},
                        "y": {"field": "y", "type": "quantitative"}
                    }
                }
            }
            if mark_type == "bar":
                # create the visualization spec dictionary
                visualization_spec_dict = raw_vega_lite_specs["bar"]
                # set the x encoding
                visualization_spec_dict["encoding"]["x"]["field"] = x_encoding
                # set the y encoding
                visualization_spec_dict["encoding"]["y"]["field"] = y_encoding
                # if available: set the color encoding
                #if num_columns == 3:
                #    visualization_spec_dict["encoding"]["color"]["field"] = color_encoding
                # check if the bin property is in the visualization spec dictionary, if so, we add a bin transform to the transform property of the vega lite spec
                # set the data property of the vega lite spec
                visualization_spec_dict["data"] = {"values": result_dataframe.to_dict(orient="records")}
            elif mark_type == "line":
                # create the visualization spec dictionary
                visualization_spec_dict = raw_vega_lite_specs["line"]
                # set the x encoding
                visualization_spec_dict["encoding"]["x"]["field"] = x_encoding
                # set the y encoding
                visualization_spec_dict["encoding"]["y"]["field"] = y_encoding
                # if available: set the color encoding
                #if num_columns == 3:
                #    visualization_spec_dict["encoding"]["color"]["field"] = color_encoding
                # check if the bin property is in the visualization spec dictionary, if so, we add a bin transform to the transform property of the vega lite spec
                # set the data property of the vega lite spec
                visualization_spec_dict["data"] = {"values": result_dataframe.to_dict(orient="records")}
            elif mark_type == "point":
                # create the visualization spec dictionary
                visualization_spec_dict = raw_vega_lite_specs["point"]
                # set the x encoding
                visualization_spec_dict["encoding"]["x"]["field"] = x_encoding
                # set the y encoding
                visualization_spec_dict["encoding"]["y"]["field"] = y_encoding
                # if available: set the color encoding
                #if num_columns == 3:
                #    visualization_spec_dict["encoding"]["color"]["field"] = color_encoding
                # check if the bin property is in the visualization spec dictionary, if so, we add a bin transform to the transform property of the vega lite spec
                # set the data property of the vega lite spec
                visualization_spec_dict["data"] = {"values": result_dataframe.to_dict(orient="records")}
            elif mark_type == "arc":
                # create the visualization spec dictionary
                visualization_spec_dict = raw_vega_lite_specs["arc"]
                # set the x encoding
                visualization_spec_dict["encoding"]["color"]["field"] = x_encoding
                # set the y encoding
                visualization_spec_dict["encoding"]["theta"]["field"] = y_encoding
                # if available: set the color encoding
                #if num_columns == 3:
                #    visualization_spec_dict["encoding"]["color"]["field"] = color_encoding
                # check if the bin property is in the visualization spec dictionary, if so, we add a bin transform to the transform property of the vega lite spec
                # set the data property of the vega lite spec
                visualization_spec_dict["data"] = {"values": result_dataframe.to_dict(orient="records")}
            # check if the bin property is in the visualization spec dictionary, if so, we add a bin transform to the transform property of the vega lite spec
            if "bin" in visualization_spec_dict:
                # get the bin axis
                bin_axis = visualization_spec_dict["bin"]["axis"]
                # get the bin type
                bin_type = visualization_spec_dict["bin"]["type"]
                # create the bin transform
                bin_transform = {"bin": {"axis": bin_axis, "type": bin_type}}
                # add the bin transform to the transform property of the vega lite spec
                visualization_spec_dict["transform"] = [bin_transform]
            # return the visualization spec dictionary
            print(visualization_spec_dict)
            return visualization_spec_dict

    def unstringify_and_unflatten_dict(self, d_str, separator='_', list_separator='-'):
        # add brackets to unbracketed string
        d_str = "{" + d_str + "}"
        # turn d_str into dictionary
        d = json.loads(d_str) # ast.literal_eval(d_str)
        ud = {}
        for k, v in d.items():
            context = ud
            keys = k.split(separator)
            for i, sub_key in enumerate(keys[:-1]):
                if list_separator in sub_key:
                    list_key, index = sub_key.split(list_separator)
                    index = int(index)
                    if list_key not in context:
                        context[list_key] = []
                    while len(context[list_key]) <= index:
                        context[list_key].append({})
                    context = context[list_key][index]
                else:
                    if sub_key not in context:
                        context[sub_key] = {}
                    context = context[sub_key]
            if list_separator in keys[-1]:
                list_key, index = keys[-1].split(list_separator)
                index = int(index)
                if list_key not in context:
                    context[list_key] = []
                while len(context[list_key]) <= index:
                    context[list_key].append({})
                context[list_key][index] = v
            else:
                context[keys[-1]] = v
        return ud

    def summary_statistics(self, df, column_name):
        """
        Returns summary statistics of a pandas dataframe column as a string.
        """
        column_dtype = df[column_name].dtype

        if column_dtype == 'object':
            # If column is categorical
            categories = df[column_name].unique().tolist()
            categories_str = '[' + ', '.join(map(str, categories)) + ']'
            return f"{column_name} ({column_dtype}) {categories_str}"

        elif column_dtype in ['int64', 'float64']:
            # If column is numerical
            min_val = df[column_name].min()
            max_val = df[column_name].max()
            avg_val = df[column_name].mean()
            return f"{column_name} ({column_dtype}) ['Min={min_val}', 'Max={max_val}', 'Avg={avg_val:.2f}']"

        elif column_dtype == 'datetime64[ns]':
            # If column is datetime
            min_date = df[column_name].min().strftime('%Y-%m-%d')
            max_date = df[column_name].max().strftime('%Y-%m-%d')
            return f"{column_name} ({column_dtype}) ['Min Date={min_date}', 'Max Date={max_date}']"
 
        else:
            # If column type is not recognized
            return "Column type not recognized"

    def add_online_training_sample(self, online_training_sample):
        #print(online_training_sample)
        # try to get the last dialog record in the online training samples db
        last_dialog_record = self.online_training_samples_db[-1]
        # add the online training sample to the turns of the last dialog record
        last_dialog_record["turns"].append(online_training_sample)
        # save the online training samples db to the online training samples db file    
        with open(self.online_training_samples_db_json_file_path, 'w') as outfile:
            json.dump(self.online_training_samples_db, outfile)
        print("Added training sample to online learning dataset.")



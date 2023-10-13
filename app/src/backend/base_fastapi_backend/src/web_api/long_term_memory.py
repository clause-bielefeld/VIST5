from abc import ABC, abstractmethod
from typing import List
from sentence_transformers import SentenceTransformer, util, evaluation
import torch
import numpy as np
import os.path
import json
import random
import pandas as pd
from web_api.tool import ToolManagerImpl

class LongTermMemory(ABC):
    pass

class LongTermMemoryImpl(LongTermMemory):
    def __init__(self, similarity_model_path: str, document_db_file_path: str = None, vector_store_file_path: str = None, fine_tune_similarity_model: bool = False):
        # self.graph_memory = [] # = KNOWLEDGE BASE in our case it is a document_db, in general: [document_db, graph_db(objects, concepts, relationships), relational_db, any_db]
        # self.skill_memory = [] # = set of policies/functions that solve input-output problems
        # fixed parameters
        self.cosine_similarity_threshold = 0.8 #0.7
        # file paths
        self.fine_tuned_similarity_model_path = './web_api/models/all-mpnet-base-v2-finetuned-'
        self.intend_db_file_path = './web_api/data/intend_db.json'
        self.tool_db_file_path = './web_api/data/tool_db.json'
        self.knowledge_db_file_path = './web_api/data/knowledge_db.json' 
        self.document_db_file_path = document_db_file_path
        self.vector_store_file_path = vector_store_file_path
        # object stores 
        self.document_db = [] # contains: intend_catalog, tool_set, knowledge_nodes
        self.example_corpus = [] # contains: intend_examples, tool_examples, knowledge_node_examples, input-output example pairs
        self.intend_db = [] # list of intend nodes = {"type":"FEW_SHOT_EXAMPLE", "intend_id", "intend_name", "intend_description", "intend_input_output_examples", "intend_instruction_template", "intend_response_template"} # intend_input_specification, intend_output_specification -> contains dataset tasks, few shot tasks, fallback intends (like e.g. greetings, goodbyes, etc.)
        self.tool_db = [] # list of tool nodes = {"type": "TOOL","tool_id", "tool_name", "tool_description", "tool__input_output_examples", "tool_instruction_template", "tool_response_template"} # tool_input_specification, tool_output_specification
        self.knowledge_db = [] # = list of knowledge_nodes = {"type": "KNOWLEDGE_NODE","knowledge_node_id", "knowledge_node_name", "knowledge_node_content", "knowledge_node_instruction_template", "knowledge_node_response_template"} 
        # load intend_catalog from file  as a list of json objects 
        with open(self.intend_db_file_path, 'r') as f:
            self.intend_db = json.load(f)
        # load tool_set from file as a list of json objects
        with open(self.tool_db_file_path, 'r') as f: 
            self.tool_db = json.load(f)
        # load knowledge_nodes from file as a list of json objects
        with open(self.knowledge_db_file_path, 'r') as f:
            self.knowledge_db = json.load(f)
        # check if document_db_file_path exists
        if os.path.isfile(self.document_db_file_path):
            # load document_db from file 
            print('... loading document_db from file')
            self.document_db = self.load_document_db(document_db_file_path)
        else:
            # generate document db from intend_catalog, tool_set, knowledge_nodes or load from file
            print('... generating document_db')
            self.document_db = self.generate_document_db()
            # save document db to json file 
            with open(self.document_db_file_path, 'w') as f:
                json.dump(self.document_db, f)
        # load semantic similarity model from similarity_model_path
        self.similarity_model = self.load_similarity_model(similarity_model_path)
        # fine tune similarity model on the document db 
        if fine_tune_similarity_model:
            print('... fine tuning similarity model')
            self.similarity_model = self.fine_tune_similarity_model()
        # check if vector_store_file_path exists, IMPORTANT: document_db has to be loaded before, because we need access to the example_corpus
        if os.path.isfile(vector_store_file_path):
            # load vector store from file
            print('... loading vector_store from file')
            self.vector_store = self.load_vector_store(vector_store_file_path)
        else:
            # create vector store from document_db using similarity model (e.g. sentence-transformers)
            print('... creating vector_store')
            self.vector_store = self.create_vector_store()
            # save vector store to file
            np.save(vector_store_file_path, self.vector_store)
        # create tool manager
        self.tool_manager = ToolManagerImpl()

    def load_document_db(self, document_db_file_path: str):
        # load document db from file
        with open(document_db_file_path, 'r') as f:
            document_db = json.load(f)
        # create example corpus from document db
        # INTENDS: load intends from document db
        input_output_examples_list = [] 
        for document in document_db:
            # get the input_output_examples list
            current_input_output_examples = document['input_output_examples']
            # append to input_output_examples_list
            input_output_examples_list.extend(current_input_output_examples)
        self.example_corpus = input_output_examples_list 
        return document_db

    def generate_document_db(self):
        # generate document db from intend_catalog, tool_set, knowledge_nodes
        document_db = []
        index_counter = 0
        for intend in self.intend_db:
            intend['index'] = index_counter 
            index_counter += 1
            document_db.append(intend)
        for tool in self.tool_db:
            tool['index'] = index_counter
            index_counter += 1
            document_db.append(tool) 
        # knowledge nodes are handled separately via TOOLS now
        #for knowledge_node in self.knowledge_db:
        #    knowledge_node['index'] = index_counter
        #    index_counter += 1
        #    document_db.append(knowledge_node)
        # create example corpus from document db
        # INTENDS: load intends from document db
        input_output_examples_list = [] 
        for document in document_db:
            # get the input_output_examples list
            current_input_output_examples = document['input_output_examples']
            # append to input_output_examples_list
            input_output_examples_list.extend(current_input_output_examples)
        # get all the answers from the input_output_examples_list
        self.example_corpus = input_output_examples_list 
        return document_db

    def load_similarity_model(self, similarity_model_path: str):
        # load semantic similarity model from similarity_model_path
        similarity_model = SentenceTransformer.load(similarity_model_path) # e.g. './models/all-mpnet-base-v2'
        return similarity_model

    def fine_tune_similarity_model(self, fine_tuning_mode='query_to_query'): # fine_tuning_mode = 'query_to_query' or 'query_to_answer' or 'query_to_query_and_answer'
        # query-to-query similarity fine tuning on tools + few shot examples
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader
        # get the model to be trained
        model = self.similarity_model
        # create a list of InputExample objects
        #Define your train examples. You need more than just two examples...
        train_examples = []
        query_query_dataset_dict = {} # python dict {"class": [{"query1":'', "query2":''}, ...], } 
        query_answer_dataset_dict = {} # python dict {"class": [{"query":'', "response":''}, ...], }
        # iterate over the document_db and create query-answer pairs
        print('... creating query-answer pairs for training')
        # INTENDS: load intends from document db
        for document in self.document_db:
            # get class
            current_type = document['type']
            # take out knowledge nodes for now
            if(current_type == 'KNOWLEDGE_NODE'):
                continue
            elif(current_type == 'FEW_SHOT_EXAMPLE'):
                # query-answer pairs = in the input_output_examples field
                current_input_output_examples = document['input_output_examples']
                for input_output_example in current_input_output_examples:
                    # get query
                    current_query = input_output_example['input']
                    # get answer
                    current_answer = input_output_example['output']
                    query_answer_object = {'query': current_query, 'answer': current_answer}
                    query_query_object = {'query1': current_query, 'query2': current_query}
                    # get intend_name -> this is used as class
                    current_intend_name = document['intend_name']
                    # append to query_answer_dataset
                    if current_intend_name in query_query_dataset_dict:
                        query_query_dataset_dict[current_intend_name].append(query_query_object)
                    else:
                        query_query_dataset_dict[current_intend_name] = [query_query_object]
                    # append to query_answer_dataset
                    if current_intend_name in query_answer_dataset_dict:
                        query_answer_dataset_dict[current_intend_name].append(query_answer_object)
                    else:
                        query_answer_dataset_dict[current_intend_name] = [query_answer_object]
            elif(current_type == 'TOOL'):
                # query-answer pairs = in the tool_input_output_examples field
                current_tool_input_output_examples = document['input_output_examples']
                for tool_input_output_example in current_tool_input_output_examples:
                    # get query
                    current_query = tool_input_output_example['input']
                    # get answer
                    current_answer = tool_input_output_example['output']
                    query_answer_object = {'query': current_query, 'answer': current_answer}
                    query_query_object = {'query1': current_query, 'query2': current_query}
                    # get tool_name -> this is used as class
                    current_tool_name = document['tool_name']
                    # append to query_query_dataset_dict 
                    if current_tool_name in query_query_dataset_dict:
                        query_query_dataset_dict[current_tool_name].append(query_query_object)
                    else:
                        query_query_dataset_dict[current_tool_name] = [query_query_object]
                    # append to query_answer_dataset_dict
                    if current_tool_name in query_answer_dataset_dict:
                        query_answer_dataset_dict[current_tool_name].append(query_answer_object)
                    else:
                        query_answer_dataset_dict[current_tool_name] = [query_answer_object]
            else:
                print('ERROR: unknown class') 
        # TRAINING SET: create training set
        # put together positive and negative pairs -> positive pairs are the query-answer pairs, negative pairs are random pairs form other classes in the list 
        # iterate over the query_query_dataset_dict and create InputExample objects
        if(fine_tuning_mode == 'query_to_query'):
            print('... creating query-query pairs for training')
            for class_name in query_query_dataset_dict:
                # get the list of query-query pairs for the current class
                current_query_query_list = query_query_dataset_dict[class_name]
                # positive pairs: iterate over the list and create InputExample objects
                for query_query_object in current_query_query_list:
                    # get query 
                    current_query1 = query_query_object['query1']
                    # SELF SIMILARITY: query to itself -> similarity  needs to be high
                    # get answer
                    current_query2 = query_query_object['query2']
                    # create InputExample object 
                    current_input_example = InputExample(texts=[current_query1, current_query2], label=1.0)
                    # append to train_examples
                    train_examples.append(current_input_example)
                    # INNER CLASS SIMILARITY:  get all the other query_query_objects in the current query list which are not the current query_query_object
                    other_positive_query_query_objects = [x for x in current_query_query_list if x != query_query_object]
                    # iterate over the other_positive_query_query_objects and create InputExample objects
                    for other_positive_query_query_object in other_positive_query_query_objects:
                        current_query2 = other_positive_query_query_object['query1']
                        # create InputExample object
                        current_input_example = InputExample(texts=[current_query1, current_query2], label=0.95)
                        # append to train_examples
                        train_examples.append(current_input_example) 
                    # OTHER CLASS DISTANCING: negative pairs: get all the other classes in the list and create several random negative InputExample objects
                    for other_class_name in query_query_dataset_dict:
                        if other_class_name != class_name:
                            # get the list of query-answer pairs for the other class 
                            other_query_query_list = query_query_dataset_dict[other_class_name]
                            # NEGATIVE HARDENING: iterate over all the other query-answer pairs and create max 5 negative InputExample objects
                            for index, other_query_query_object in enumerate(other_query_query_list):
                                # max index -> we want a maximum of 5 negative pairs per positive pair
                                if(index > 4):
                                    break
                                # query = current_query -> because we want the similarity to be low
                                # get answer
                                other_query = other_query_query_object['query1']
                                # create InputExample object
                                other_input_example = InputExample(texts=[current_query1, other_query], label=0.05)
                                # append to train_examples
                                train_examples.append(other_input_example)
        elif(fine_tuning_mode == 'query_to_answer'):
            print('... creating query-answer pairs for training')
            for class_name in query_answer_dataset_dict:
                # get the list of query-answer pairs for the current class
                current_query_answer_list = query_answer_dataset_dict[class_name]
                # positive pairs: iterate over the list and create InputExample objects
                for query_answer_object in current_query_answer_list:
                    # get query 
                    current_query = query_answer_object['query']
                    # get answer
                    current_answer = query_answer_object['answer']
                    # create InputExample object
                    current_input_example = InputExample(texts=[current_query, current_answer], label=1.0)
                    # append to train_examples
                    train_examples.append(current_input_example)
                    # negative pairs: get all the other classes in the list and create one random negative InputExample objects
                    for other_class_name in query_answer_dataset_dict:
                        if other_class_name != class_name:
                            # get the list of query-answer pairs for the other class 
                            other_query_answer_list = query_answer_dataset_dict[other_class_name]
                            # get a random query-answer pair from the other class
                            other_query_answer_object = random.choice(other_query_answer_list)
                            # query = current_query -> because we want the similarity to be low
                            # get answer
                            other_answer = other_query_answer_object['answer']
                            # create InputExample object
                            other_input_example = InputExample(texts=[current_query, other_answer], label=0.1)
                            # append to train_examples
                            train_examples.append(other_input_example)
        elif(fine_tuning_mode == 'query_to_query_and_answer'):
            print('... creating query-query and query-answer pairs for training')
            for class_name in query_query_dataset_dict:
                # get the list of query-query pairs for the current class
                current_query_query_list = query_query_dataset_dict[class_name]
                # positive pairs: iterate over the list and create InputExample objects
                for query_query_object in current_query_query_list:
                    # get query 
                    current_query1 = query_query_object['query1']
                    # get answer
                    current_query2 = query_query_object['query2']
                    # create InputExample object
                    current_input_example = InputExample(texts=[current_query1, current_query2], label=1.0)
                    # append to train_examples
                    train_examples.append(current_input_example)
                    # negative pairs: get all the other classes in the list and create one random negative InputExample objects
                    for other_class_name in query_query_dataset_dict:
                        if other_class_name != class_name:
                            # get the list of query-answer pairs for the other class 
                            other_query_query_list = query_query_dataset_dict[other_class_name]
                            # get a random query-answer pair from the other class
                            other_query_query_object = random.choice(other_query_query_list)
                            # query = current_query -> because we want the similarity to be low
                            # get answer
                            other_query = other_query_query_object['query1']
                            # create InputExample object
                            other_input_example = InputExample(texts=[current_query, other_query], label=0.1)
                            # append to train_examples
                            train_examples.append(other_input_example)
            for class_name in query_answer_dataset_dict:
                # get the list of query-answer pairs for the current class
                current_query_answer_list = query_answer_dataset_dict[class_name]
                # positive pairs: iterate over the list and create InputExample objects
                for query_answer_object in current_query_answer_list:
                    # get query 
                    current_query = query_answer_object['query']
                    # get answer
                    current_answer = query_answer_object['answer']
                    # create InputExample object
                    current_input_example = InputExample(texts=[current_query, current_answer], label=1.0)
                    # append to train_examples
                    train_examples.append(current_input_example)
                    # negative pairs: get all the other classes in the list and create one random negative InputExample objects
                    for other_class_name in query_answer_dataset_dict:
                        if other_class_name != class_name:
                            # get the list of query-answer pairs for the other class 
                            other_query_answer_list = query_answer_dataset_dict[other_class_name]
                            # get a random query-answer pair from the other class
                            other_query_answer_object = random.choice(other_query_answer_list)
                            # query = current_query -> because we want the similarity to be low
                            # get answer
                            other_answer = other_query_answer_object['answer']
                            # create InputExample object
                            other_input_example = InputExample(texts=[current_query, other_answer], label=0.1)
                            # append to train_examples
                            train_examples.append(other_input_example)
        # TRAINING: fine tune the model
        print('... finetuning the similarity model.')
        # create a DataLoader for the train examples    
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        # define the loss function
        train_loss = losses.CosineSimilarityLoss(model=model) 
        # train the model
        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=2) # 5 epochs? warmup_steps=20
        # EVALUATION: evaluate the model on training data, for test reasons only to know how well the model is doing after n epochs
        # create an evaluator object using the EmbeddingSimilarityEvaluator class 
        sentences1 = [example.texts[0] for example in train_examples] 
        sentences2 = [example.texts[1] for example in train_examples]  
        scores = [example.label for example in train_examples]
        evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1=sentences1,sentences2=sentences2,scores=scores)
        # evaluate the model on the training data
        train_score = evaluator(model)
        print(f"Evaluation Score on train set: {train_score:.4f}") 
        # save the model
        model_saving_path = self.fine_tuned_similarity_model_path + fine_tuning_mode 
        model.save(model_saving_path)  
        print('... similarity model fine-tuned.')
        return model  

    def create_vector_store(self):
        # create vector store from document_db using QUERY similarity model (e.g. sentence-transformers) 
        input_examples_list = [input_output_example['input'] for input_output_example in self.example_corpus]
        # for making the retrieval effective, we need to find the most similar query in our corpus to the given query by the user -> take all the example queries from the document_db and encode them using the similarity model
        vector_store = self.similarity_model.encode(input_examples_list, normalize_embeddings=True, show_progress_bar=True) # corpus_embeddings come back as a numpy array
        return vector_store

    def load_vector_store(self, vector_store_file_path: str):
        # load vector store from vector_store_file_path
        vector_store = np.load(vector_store_file_path)
        return vector_store 

    def retrieve_examples_from_knowledge_base(self, query: str, k: int = 3):
        # find the most similar document in the document_db to the query
        # first encode the query using the similarity model
        query_embedding = self.similarity_model.encode(query)
        # then find the top k most similar document to the query by comparing the query vector to the document vectors in the vector store
        corpus_embeddings = self.vector_store
        top_k_examples = util.semantic_search(query_embedding, corpus_embeddings, score_function=util.cos_sim, top_k=k) # {'corpus_id', 'score'}
        top_k_examples = top_k_examples[0]  
        # THRESHOLD: for every example in the top_k_examples, check if it is >= self.cosine_similarity_threshold, if not, remove it from the list
        top_k_examples = [example for example in top_k_examples if example['score'] >= self.cosine_similarity_threshold]
        # get the example ids of the top k examples
        top_k_example_ids = [example['corpus_id'] for example in top_k_examples]
        print(top_k_example_ids)
        # get the examples from the self.example_corpus using the top_k_example_ids
        top_k_examples = [self.example_corpus[example_index] for example_index in top_k_example_ids]
        return top_k_examples 

    def retrieve_documents_from_knowledge_base(self, query: str, k: int = 3):
        # find the most similar document in the document_db to the query
        # first encode the query using the similarity model
        #query_embedding = self.similarity_model.encode(query)
        # then find the top k most similar document to the query by comparing the query vector to the document vectors in the vector store
        #corpus_embeddings = self.vector_store
        #top_k_documents = util.semantic_search(query_embedding, corpus_embeddings, score_function=util.cos_sim, top_k=k) # {'corpus_id', 'score'}
        #top_k_documents = top_k_documents[0]
        # THRESHOLD: for every document in the top_k_documents, check if it is >= self.cosine_similarity_threshold, if not, remove it from the list
        #top_k_documents = [document for document in top_k_documents if document['score'] >= self.cosine_similarity_threshold]
        # get the document ids of the top k documents
        #top_k_document_ids = [document['corpus_id'] for document in top_k_documents]
        # get the documents from the self.document_db using the top_k_document_ids
        #top_k_documents = [self.document_db[i] for i in top_k_document_ids]
        top_k_documents = [] # not implemented yet.
        return top_k_documents

    def get_document_from_knowledge_base(self, document_id: int):
        # the document id is the index of the document in the document_db list
        document = self.document_db[document_id]
        return document

    def get_document_type_from_knowledge_base(self, document_id: int):
        # the document id is the index of the document in the document_db list
        document = self.document_db[document_id]
        # the document type is the stored in the "type" field of the document
        document_type = document["type"]
        return document_type
    
    def check_if_tool_function_exists(self, function_name: str):
        # check if a function exists in the tool_db 
        valid_functions = [tool['tool_function_name'] for tool in self.tool_db]
        if function_name in valid_functions:
            return True
        else:
            return False

    def get_function_from_tool_db(self, function_name: str):
        # get the function from the tool_db
        function = [tool for tool in self.tool_db if tool['tool_function_name'] == function_name][0]
        return function
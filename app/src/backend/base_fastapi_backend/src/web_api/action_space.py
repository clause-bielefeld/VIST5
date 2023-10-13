from abc import ABC, abstractmethod
from typing import List
import pandas as pd
import json
import sqlite3
import ast
import random
from web_api.utils.vega_zero_to_vega_lite import VegaZero2VegaLite

class ActionSpace(ABC):
    pass

class ActionSpaceImpl(ActionSpace):
    def __init__(self, agent):
        self.agent = agent # sets the reference to the agent, such that the action space can update the agents state
        self.action_list = ["sql_tool", "search_tool", "update_vis", "create_vis", "text_response", "create_vega", "create_vega_lite"]
        self.internal_loop_actions = ["sql_tool", "search_tool"]
        # create external loop actions as the complement of the internal loop actions and the action list
        self.direct_response_actions = list(set(self.action_list) - set(self.internal_loop_actions)) # =  ["update_vis", "create_vis", "text_response"]
        # TOOLS 
        self.sql_connection = sqlite3.connect(':memory:')
        self.max_sql_result_rows = 10
        self.vega_zero_to_vega_lite_converter = VegaZero2VegaLite()
        # UTILS
        self.output_texts_list = ["Sure thing! I have created a plot based on your query.", 
                     'No worries! I just updated your plot as you asked.', 
                     'No problem! I made the necessary changes to your plot.',
                     'Of course! I went ahead and updated your plot based on your request.', 
                     'Absolutely! I took care of updating your plot according to your request.',
                     'Sure thing! I just completed the changes to your plot as requested.', 
                     'You got it! I just finished updating your plot according to your instructions.',
                     'You bet! I have modified your plot based on your instructions.', 
                     'You got it! I have updated your plot to meet your needs.', 
                     'You got it! I have made the changes to your plot as requested.', 
                     'No problem! I just made the changes to your plot according to your request.']
        self.host_url = "http://localhost:8080" #"https://nlpvisualizationevaluation.uni-jena.de" # "http://localhost:8080"
        self.frontend_table_files_base_path = self.host_url + "/static/ui/pages/web_pages/playgrounds/vegalite_playground/assets/tables/"
        # LOAD FEW SHOT ACTIONS: 
        self.load_few_shot_actions()

    def get_action_list(self): 
        return self.action_list
    
    def get_internal_loop_actions(self):
        return self.internal_loop_actions

    def get_direct_response_actions(self):
        return self.direct_response_actions

    def load_few_shot_actions(self):
        # load the few shot actions from the examples_corpus of the long term memory of the agent
        examples_corpus = self.agent.memory.long_term_memory.example_corpus
        # iterate over the examples corpus and load the few shot actions
        for example in examples_corpus:
            # get the output of the example
            example_output = example["output"]
            # from the output, get the action, format looks like this: "action: locomotion; args: text=Sure, here you go|type=move_right;"
            action_name = example_output.split(";")[0].replace("action: ", "").strip()
            # check if this class has a method with the name of the action
            if hasattr(self, action_name):
                # check if the action is in the action list already
                if action_name not in self.action_list: 
                    # add the action to the action list
                    self.action_list.append(action_name) 
            else: 
                print("WARNING: action '{}' is not implemented in the action space.".format(action_name))
        # update the direct response actions
        self.direct_response_actions = list(set(self.action_list) - set(self.internal_loop_actions)) 
                
    # IMPLEMENT ALL ACTIONS HERE 
    ## TOOL ACTIONS
    def sql_tool(self, args_flattened_str: str):
        """
        SQL TOOL: either pandasql or sqlite + pandas: https://towardsdatascience.com/an-easy-beginners-guide-to-sqlite-in-python-and-pandas-fbf1f38f6800
        :param sql_query: the sql query
        :return: result dataframe containing the result of the sql query
        """
        # get sql_query from args_list
        # unflatten dictionary
        args_dict = self.agent.unstringify_and_unflatten_dict(args_flattened_str)
        sql_query = args_dict["sql_query"]
        # execute sql query on the sql_connection -> dataframe has been set to sqlite connection in update_table function of agent already
        self.agent.current_result_dataframe = pd.read_sql_query(sql_query, self.sql_connection)
        print(self.agent.current_result_dataframe.head())
        # turn the result dataframe into a a list of rows (each row is a list of values)
        sql_result_list = []
        # first list in the sql result list is the column names
        column_names = self.agent.current_result_dataframe.columns.tolist()
        sql_result_list.append(column_names)
        # iterate over the rows in the dataframe and append them to the sql result list
        for index, row in self.agent.current_result_dataframe.iterrows():
            # check if we have reached the maximum number of rows
            if index >= self.max_sql_result_rows:
                break
            row_list = row.tolist()
            sql_result_list.append(row_list)
        tool_output = self.sql_result_to_string(sql_result_list)
        action_name = "sql_tool"
        args_list = [tool_output]
        return action_name, args_list

    def search_tool(self, args_flattened_str: str):
        """
        SEARCH TOOL: we use wikipedia as search tool example here
        :param search_query: the search query
        :return: list of strings containing the search results
        """
        pass

    ## NON-TOOL ACTIONS
    def update_vis(self, args_flattened_str: str): ### lEGACY CODE
        """
        UPDATE VIS: performs the execution of the update vis action
        :param changes_dictionary: the changes dictionary
        :param text_output: the text output
        :return: the updated visualization, which is a vega_lite_spec dictionary + the text output
        """
        frontend_args_list = [] # list of arguments to be returned to the frontend
        # the update action has a text argument and an vega_update argument
        # iterate over the args list and find the text argument, make sure to strip the whitespaces before and after the text
        text_arg = "Plot updated."
        frontend_args_list.append(text_arg)
        # get all vega update arguments 
        args_dict = self.agent.unstringify_and_unflatten_dict(args_flattened_str)
        change_dictionary = args_dict
        # update the visualization state
        try:
            visualization_state_dict = self.agent.memory.short_term_memory.graph_memory['visualization_state']
            visualization_state_dict = self.agent.update_dictionary(visualization_state_dict, change_dictionary) 
            #print(visualization_state_dict)
            # update visualization state
            self.agent.memory.short_term_memory.graph_memory['visualization_state'] = visualization_state_dict
        except Exception as e:
            print("ERROR: Could not update visualization state." + str(key) + " " + str(value))
            print(e)
        # after updating the visualization state, get the current visualization state as json string and add it to the args list
        visualization_state_dict = self.agent.memory.short_term_memory.graph_memory['visualization_state']
        visualization_state_json = json.dumps(visualization_state_dict)
        frontend_args_list.append(visualization_state_json)
        # return the action name and args_list containing the text argument and the visualization state json string
        action_name = "update_vis"
        return action_name, frontend_args_list

    def create_vis(self, args_list: List[str]): # LEGACY CODE
        """
        CREATE VIS: performs the execution of the create vis action 
        :param text_output: the output text to the user
        :param vis_spec_str: the visualization specification
        :return: the created visualization, which is a vega_lite_spec dictionary + a randomly selected text response
        """
        frontend_args_list = [] 
        # the create action has a text argument and a vega_zero argument 
        # iterate over the args list and find the text argument, make sure to strip the whitespaces before and after the text
        for arg in args_list: 
            #print("arg: ", arg)
            if "text=" in arg:
                text_arg = arg.replace("text=", "").strip()
                # append the text argument to the args_list
                frontend_args_list.append(text_arg)
            elif "vis_spec=" in arg:
                # if we have a visualization spec -> we create a novel vega lite visualization from that spec
                # if we do not have a visualization spec, we use a recommender engine to select the best possible vega lite visualization for the given result dataframe
                vis_spec_string = arg.replace("vis_spec=", "").strip()
                # create the visualization state dictionary from the vis_spec_string argument
                try:
                    results_dataframe = self.agent.current_result_dataframe # has been updated by the sql tool before, otherwise create_vis cannot be called
                    new_vega_lite_dict = self.agent.create_visualization_from_spec(vis_spec_string, results_dataframe)
                    self.agent.memory.short_term_memory.graph_memory['visualization_state'] = new_vega_lite_dict
                    print(new_vega_lite_dict)
                except Exception as e:
                    print("ERROR: could not convert vis_spec to vega_lite_dict: ", e)
                    return "text_response", ["I am sorry, I did not understand your query. Maybe you can rephrase it?"]
        # after updating the visualization state, get the current visualization state as json string and add it to the args list
        visualization_state_dict = self.agent.memory.short_term_memory.graph_memory['visualization_state']
        visualization_state_json = json.dumps(visualization_state_dict)
        frontend_args_list.append(visualization_state_json) 
        # add results dataframe to the args list as html string using pandas to html 
        results_dataframe = self.agent.current_result_dataframe
        # convert the dataframe to html string but without the index column
        results_dataframe_html = results_dataframe.to_html(index=False, classes="table is-striped is-fullwidth")
        frontend_args_list.append(results_dataframe_html)
        #print("frontend_args_list: ", frontend_args_list)
        # return the action name and the args list containing the text 
        action_name = "create_vis"
        return action_name, frontend_args_list

    def create_vega(self, args_list: List[str]): # LEGACY CODE
        frontend_args_list = [] 
        for arg in args_list: 
            if "vega_zero=" in arg:
                vega_zero_string = arg.replace("vega_zero=", "").strip()
                try: 
                    results_dataframe = self.agent.current_dataframe # here we have no sql query on the data -> so the results dataframe is the normal dataframe
                    new_vega_lite_dict = self.vega_zero_to_vega_lite_converter.to_VegaLite(vega_zero_string, results_dataframe)
                    self.agent.memory.short_term_memory.graph_memory['visualization_state'] = new_vega_lite_dict
                except Exception as e:
                    print("ERROR: could not convert vega_zero to vega_lite_dict: ", e)
                    return "text_response", ["I am sorry, I did not understand your query. Maybe you can rephrase it?"]
        # create a text response and add it to the args list
        output_text = random.choice(self.output_texts_list) 
        frontend_args_list.append(output_text)
        # after updating the visualization state, get the current visualization state as json string and add it to the args list
        visualization_state_dict = self.agent.memory.short_term_memory.graph_memory['visualization_state']
        # replace the 'data' key with the real path of the data file 
        visualization_state_dict['data'] = {"name": str(self.agent.current_table_name), "url": str(self.frontend_table_files_base_path + self.agent.current_table_name + ".csv")}
        visualization_state_json = json.dumps(visualization_state_dict)
        frontend_args_list.append(visualization_state_json) 
        #print("frontend_args_list: ", frontend_args_list)
        action_name = "create_vega"
        # return the action name and the args list containing the text 
        return action_name, frontend_args_list   

    def create_vega_lite(self, args_flattened_str: str):
        try:
            frontend_args_list = [] 
            # unflatten and unstringify the vega lite string 
            vega_lite_flattened_string = args_flattened_str
            new_vega_lite_dict = self.agent.unstringify_and_unflatten_dict(vega_lite_flattened_string)
            #print(new_vega_lite_dict)
            self.agent.memory.short_term_memory.graph_memory['visualization_state'] = new_vega_lite_dict
            # create a text response and add it to the args list
            output_text = random.choice(self.output_texts_list) 
            frontend_args_list.append(output_text)
            # after updating the visualization state, get the current visualization state as json string and add it to the args list
            visualization_state_dict = self.agent.memory.short_term_memory.graph_memory['visualization_state']
            # replace the 'data' key with the real path of the data file 
            visualization_state_dict['data'] = {"name": str(self.agent.current_table_name), "url": str(self.frontend_table_files_base_path + self.agent.current_table_name + ".csv")}
            visualization_state_json = json.dumps(visualization_state_dict)
            frontend_args_list.append(visualization_state_json) 
            #print("frontend_args_list: ", frontend_args_list)
            action_name = "create_vega_lite"
            # return the action name and the args list containing the text 
        except Exception as e:
            print("ERROR: could not convert flattened vega_lite string to vega_lite_dict: ", e)
            print(args_flattened_str)
            frontend_args_list = ["I am sorry, I did not understand your query. Maybe you can rephrase it?"]
            action_name = "text_response"
        return action_name, frontend_args_list   

    def text_response(self, args_flattened_str: str):
        """
        TEXT RESPONSE: performs the execution of the text response action
        :param text_output: the text output
        :return: the text output
        """
        frontend_args_list = []
        try:
            args_dict = self.agent.unstringify_and_unflatten_dict(args_flattened_str) 
            text_arg = args_dict['text']
            frontend_args_list.append(text_arg)
        except Exception as e:
            print("ERROR: could not convert flattened text string to text string: ", e)
            frontend_args_list.append("I am sorry, I did not understand your query. Maybe you can rephrase it?")
        action_name = "text_response"
        return action_name, frontend_args_list

    ## CUSTOM ACTIONS/FUNCTIONS/INTENDS
    def download_visualization(self, args_flattened_str: str):
        """
        DOWNLOAD VISUALIZATION: sends a signal to the frontend to activate the download of the current vega lite chart
        """
        frontend_args_list = []
        text_arg = "Activating download of the current visualization."
        frontend_args_list.append(text_arg)
        action_name = "download_visualization"
        return action_name, frontend_args_list

    def navigate_city(self, args_flattened_str: str):
        frontend_args_list = []
        try:
            args_dict = self.agent.unstringify_and_unflatten_dict(args_flattened_str)
            text_arg = "Navigating to the city of " + args_dict['city'] + "."
            frontend_args_list.append(text_arg)
            city_arg = args_dict['city']
            frontend_args_list.append(city_arg)
            action_name = "navigate_city" 
        except Exception as e:
            print("ERROR: could not convert flattened city dict to string: ", e)
            frontend_args_list = ["I am sorry, I did not understand your query. Maybe you can rephrase it?"]
            action_name = "text_response"
        # return action name and frontend args list
        return action_name, frontend_args_list

    def locomotion(self, args_flattened_str: str):
        print('locomotion called')
        frontend_args_list = []
        try:
            args_dict = self.agent.unstringify_and_unflatten_dict(args_flattened_str)
            text_arg = "Executing locomotion action of type " + args_dict['type'] + "."
            frontend_args_list.append(text_arg)
            locomotion_type = args_dict['type']
            frontend_args_list.append(locomotion_type)
            action_name = "locomotion"
        except Exception as e:
            print("ERROR: could not convert flattened locomotion dict to string: ", e)
            frontend_args_list = ["I am sorry, I did not understand your query. Maybe you can rephrase it?"]
            action_name = "text_response"
        # return action name and frontend args list
        return action_name, frontend_args_list

    def change_map(self, args_flattened_str: str):
        frontend_args_list = []
        try:
            args_dict = self.agent.unstringify_and_unflatten_dict(args_flattened_str)
            text_arg = "Changing map type to " + args_dict['type'] + "."
            frontend_args_list.append(text_arg)
            map_type = args_dict['type']
            frontend_args_list.append(map_type)
            action_name = "change_map_type"
        except Exception as e:
            print("ERROR: could not convert flattened map type dict to string: ", e)
            frontend_args_list = ["I am sorry, I did not understand your query. Maybe you can rephrase it?"]
            action_name = "text_response"
        # return action name and frontend args list
        return action_name, frontend_args_list

    def marker_plot(self, args_flattened_str: str):
        frontend_args_list = []
        try:
            args_dict = self.agent.unstringify_and_unflatten_dict(args_flattened_str)
            text_arg = "Plotting markers for dataset " + str(self.agent.current_table_name) + "."
            frontend_args_list.append(text_arg)
            # get the distinct lat, lon values of all locations in the current dataframe
            current_dataframe = self.agent.current_dataframe
            # get the distinct lat, lon value pairs in the dataframe
            lat_lon_values = current_dataframe[['latitude', 'longitude']].drop_duplicates().values.tolist()
            print("lat_lon_values: ", lat_lon_values)
            # add the lat, lon values to the frontend args list
            frontend_args_list.append(lat_lon_values)
            action_name = "marker_plot"
        except Exception as e:
            print("ERROR: could not finish marker plot action: ", e)
            frontend_args_list = ["I am sorry, I did not understand your query. Maybe you can rephrase it?"]
            action_name = "text_response"
        # return action name and frontend args list
        return action_name, frontend_args_list

    def heat_map(self, args_flattened_str: str):
        frontend_args_list = []
        try:
            args_dict = self.agent.unstringify_and_unflatten_dict(args_flattened_str)
            text_arg = "Plotting heat map for dataset " + "northern_european_cities" + " and column " + str(args_dict['column']) + "." #  str(self.agent.current_table_name) 
            frontend_args_list.append(text_arg)
            column_arg = args_dict['column']
            # get the distinct lat, lon values of all locations in the current dataframe
            current_weather_df = self.get_current_weather_dataframe()
            # get the intensity map from the current weather dataframe as latitude, longitude, column_arg value
            intensity_map = current_weather_df[['latitude', 'longitude', column_arg]].values.tolist()
            #print("intensity_map: ", intensity_map)
            # add the lat, lon values to the frontend args list
            frontend_args_list.append(intensity_map)
            action_name = "heat_map" 
        except Exception as e:
            print("ERROR: could not finish heat map action: ", e)
            frontend_args_list = ["I am sorry, I did not understand your query. Maybe you can rephrase it?"]
            action_name = "text_response"
        # return action name and frontend args list 
        return action_name, frontend_args_list

    def flow_map(self, args_flattened_str: str):
        frontend_args_list = []
        try:
            args_dict = self.agent.unstringify_and_unflatten_dict(args_flattened_str)
            text_arg = "Plotting flow map for dataset " + "northern_european_cities.csv" + "."
            frontend_args_list.append(text_arg)
            #column_arg = args_dict['column']
            # get the distinct lat, lon values of all locations in the current dataframe
            #current_dataframe = self.agent.current_dataframe
            # get the distinct lat, lon value pairs in the dataframe
            #lat_lon_values = current_dataframe[['latitude', 'longitude']].drop_duplicates().values.tolist()
            #print("lat_lon_values: ", lat_lon_values)
            # add the lat, lon values to the frontend args list
            wind_map = self.get_current_wind_map()
            frontend_args_list.append(wind_map)
            action_name = "flow_map"
        except Exception as e:
            print("ERROR: could not finish flow map action: ", e)
            frontend_args_list = ["I am sorry, I did not understand your query. Maybe you can rephrase it?"]
            action_name = "text_response"
        # return action name and frontend args list
        return action_name, frontend_args_list

    def reset_map(self, args_flattened_str: str):
        frontend_args_list = []
        try:
            args_dict = self.agent.unstringify_and_unflatten_dict(args_flattened_str)
            text_arg = "Resetting leaflet map."
            frontend_args_list.append(text_arg)
            action_name = "reset_map"
        except Exception as e:
            print("ERROR: could not finish reset map action: ", e)
            frontend_args_list = ["I am sorry, I did not understand your query. Maybe you can rephrase it?"]
            action_name = "text_response"
        # return action name and frontend args list
        return action_name, frontend_args_list
        
    ## UTILITY FUNCTIONS
    def sql_result_to_string(self, sql_result_list, max_rows=10):
        # check if sql_result is an empty list
        if not sql_result_list:
            return ""
        else:
            # get column names
            col_names = sql_result_list[0]
            # get rows
            rows = sql_result_list[1:]
            # limit number of rows to max_rows
            rows = rows[:self.max_sql_result_rows]
            # check if a row exists that is None, if so remove the row from the list
            rows = [row for row in rows if row is not None]
            # turn all elements into strings
            rows = [[str(element) for element in row] for row in rows]
            # create string
            string = "col : " + " | ".join(col_names) + " "
            # add rows
            for i, row in enumerate(rows):
                string += "row {} : ".format(i) + " | ".join(row) + " "
            return string
 
    def get_current_wind_map(self):
        # load current wind map from './web_api/datasets/wind.json' into a python list 
        with open('./web_api/data/datasets/wind.json') as f:
            wind_map = json.load(f) 
        return wind_map 

    def get_current_weather_dataframe(self):
        # load current weather data from './web_api/tables/current_weather.csv' into a pandas dataframe
        current_weather_df = pd.read_csv('./web_api/data/tables/current_weather.csv')
        return current_weather_df
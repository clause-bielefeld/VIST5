from abc import ABC, abstractmethod
from web_api.utils.vega_zero_to_vega_lite import VegaZero2VegaLite

class Tool(ABC):
    pass 

#class WikipediaSearchTool(Tool):
#    def __init__(self):
#        pass
#
#    def wikipedia_search_tool(self, search_query: str):
#        # take the search query and call the google search api

class ToolManager(ABC):
    pass 

class ToolManagerImpl(ToolManager):
    def __init__(self):
        # initialize vega zero to vega lite converter
        self.vega_zero_to_vega_lite_converter = VegaZero2VegaLite()

    # TOOL FUNCTIONS --------------------------------------------
    def search_tool(self, search_query: str):
        # take the search query and call the google search api
        # example: output_string = "```wikipedia_search_tool(What is the capital of Germany?)```"
        # TODO:
        print('search_query: ', search_query)
        response = "WIKIPEDIA SEARCH TOOL: This is a test response."
        # return the response as a string so that it can be used to populate the context
        return response

    # BACKEND FUNCTIONS --------------------------------------------
    def create_visualization(self, vega_zero_string: str):
        # convert the vega zero string to a vega lite dict
        vega_lite_dict = self.vega_zero_to_vega_lite_converter.to_VegaLite(vega_zero_string) 
        # update visualization state in short term memory -> get the short term memory object
        # TODO:
        frontend_func_name = "update_visualization"
        frontend_func_args = [vega_lite_dict]
        return frontend_func_name, frontend_func_args

    def update_visualization(self, args_to_update_dict: dict):
        # get the vega lite specs to update from the dict/json object??? TODO: define the form here, also in the training data
        # update the visualization state in short term memory
        # TODO: 
        # get updated visualization state
        updated_visualization_state = {}
        frontend_func_name = "update_visualization"
        frontend_func_args = updated_visualization_state
        return frontend_func_name, frontend_func_args

    


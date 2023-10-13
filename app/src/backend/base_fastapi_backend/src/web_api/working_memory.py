from abc import ABC, abstractmethod
from typing import List

class WorkingMemory(ABC):
    pass

class WorkingMemoryImpl(WorkingMemory):
    def __init__(self):
        self.personality_prompt_template = PERSONALITY_PROMPT_TEMPLATE
        # open dialog prompt templates
        self.open_dialog_prompt_template = OPEN_DIALOG_PROMPT_TEMPLATE
        # retrieval augmented dialog prompt templates
        self.base_prompt_template= BASE_PROMPT_TEMPLATE
        self.user_input_prompt_template = USER_INPUT_PROMPT_TEMPLATE
        self.tool_input_prompt_template = TOOL_INPUT_PROMPT_TEMPLATE


# --------------------------------------------
# PROMPT TEMPLATES

USER_INPUT_PROMPT_TEMPLATE = """INPUT: User: {user_input}{examples}{context}"""
# usage:
#examples = ["example1", "example2"]
#context = "This is the context."
# check if examples is non-empty
#examples_str = "Here are some examples:\n{}\n".format("\n".join(examples)) if examples else ""
# check if context is non-empty
#context_str = "Context: {}\n".format(context) if context else ""
#prompt = USER_INPUT_PROMPT_TEMPLATE.format(user_input="Alice", examples_str=examples_str, context_str=context_str)

TOOL_INPUT_PROMPT_TEMPLATE = """INPUT: Observation: {tool_output}"""

BASE_PROMPT_TEMPLATE = """
STATE:  
{state_prompt}
=========

Conversation History:
{chat_history}

NEW {input_prompt}
OUTPUT:"""


# --------------------------------------------

OPEN_DIALOG_PROMPT_TEMPLATE = """ 
{personality_prompt}

Current conversation: 
{chat_history}
INPUT: {user_input}
OUTPUT:"""
# --------------------------------------------



# --------------------------------------------
# LEGACY PROMPT TEMPLATES

PERSONALITY_PROMPT_TEMPLATE = """VIST5 is a large language model trained for visualization-oriented dialogue tasks. It generates visualizations from natural language descriptions, modifies them based on user commands and answers any questions users may have truthfully and reliable."""

INNER_MONOLOG_PROMPT_TEMPLATE = """
{chat_history}
USER: {user_input}
Thought: Do I need more context to generate a response?  
"""
# => output here would then be: yes/no

LEG_CONTEXT_PROMPT_TEMPLATE = """ 
EXAMPLES: 
{examples}
=========
CONTEXT:
{context}
=========
STATE:  
{state_prompt}
=========

Current conversation: 
{chat_history}
USER: {user_input}
VIST5:"""

LEGACY_CONTEXT_PROMPT_TEMPLATE = """ 
EXAMPLES: 
{examples}
=========
CONTEXT:
{context}
=========
STATE: 
{state_prompt}
=========

TASK: 
Given the EXAMPLES, CONTEXT and the STATE of the visualization above, create a response to the USER input in the conversation below.
If the EXAMPLES and the CONTEXT are not helpful or if you do not know the answer, just truthfully say you do not know.

Current conversation: 
{chat_history}
USER: {user_input}
VIST5:"""

OLD_CONTEXT_PROMPT_TEMPLATE = """
{personality_prompt}

=========  
CONTEXT: 
{context}

=========
STATE: 
{state_prompt}
=========

TASK: 
Given the CONTEXT and the STATE of the visualization above, create a response to the USER input in the conversation below.
If the CONTEXT is not helpful or if you do not know the answer, just truthfully say you do not know.

Current conversation: 
{chat_history}
USER: {user_input}
VIST5:"""

# old personality prompt template
"""
VIST5 is designed to be able to assist with tasks like question answering, explanation generation, visualization modification, and visualization generation. 
As a language model, VIST5 is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand. 
VIST5 is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. 
Additionally, AI is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics. 
Overall, VIST5 is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. 
Whether you need help with a specific question or just want to have a conversation about a particular topic, AI is here to assist.
"""
# --------------------------------------------



# 3rd party imports
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from web_api.dialog_manager import DialogManagerImpl
from web_api.agent import AgentImpl
import os
import time

# Using https://fastapi.tiangolo.com/ for the backend
app = FastAPI()

# Add static HTML,CSS,JS and use template engine to create docs for the API
app.mount("/static", StaticFiles(directory="web_frontend/"), name="static")
templates = Jinja2Templates(directory="web_frontend/")

@app.on_event("startup") 
async def startup_event():
    print('Starting up the server')


@app.on_event("shutdown")
def shutdown_event():
    print('Application shutdown.')
    # with open("log.txt", mode="a") as log:
    #    log.write("Application shutdown")

# ----------------------------------------------------------------------
# REQUEST MODELS 
class TextRequest(BaseModel):  
    text: str   

# ----------------------------------------------------------------------
# OBJECTS
document_db_file_path = './web_api/data/document_db.json' # to renew the document db, delete the file and restart the server
vector_store_file_path = './web_api/data/vector_store.npy' # to renew the vector store, delete the file and restart the server
similarity_model_path = './web_api/models/all-mpnet-base-v2-finetuned-query_to_query' #'./web_api/models/all-mpnet-base-v2' #./web_api/models/all-mpnet-base-v2' # './web_api/models/all-mpnet-base-v2-finetuned' # './web_api/models/all-MiniLM-L6-v2' # './web_api/models/all-MiniLM-L6-v2-finetuned'
fine_tune_similarity_model = False #True          
llm_path = './web_api/models/vist5_base_model_onnx' #'./web_api/models/vist5_base_model'  #'./web_api/models/flan_t5_base' 
llm_tokenizer_path = './web_api/models/vist5_base_tokenizer' #'./web_api/models/flan_t5_base_tokenizer'
 
dialog_agent = AgentImpl(llm_path=llm_path, llm_tokenizer_path=llm_tokenizer_path, similarity_model_path=similarity_model_path, fine_tune_similarity_model=fine_tune_similarity_model, document_db_file_path=document_db_file_path, vector_store_file_path=vector_store_file_path)
 
# ----------------------------------------------------------------------
# ROUTES 
@app.get('/') 
async def get_single_page_app(request: Request):
    return templates.TemplateResponse("ui/pages/web_pages/single_page_app/single_page_app.html", {"request": request})

@app.get("/vegalite_playground", response_class=HTMLResponse)
async def keywordscape_playground(request: Request):
    return templates.TemplateResponse("ui/pages//web_pages/playgrounds/vegalite_playground/vegalite_playground.html", {"request": request})

# ---------------------------------------------------------------------------------------------
@app.get("/reset_dialog_agent")
async def reset_dialog_agent():
    global dialog_agent
    global llm_path
    global llm_tokenizer_path
    global similarity_model_path
    global fine_tune_similarity_model
    global document_db_file_path
    global vector_store_file_path
    # delete old dialog agent
    del dialog_agent
    # create new dialog agent
    dialog_agent = AgentImpl(llm_path=llm_path, llm_tokenizer_path=llm_tokenizer_path, similarity_model_path=similarity_model_path, fine_tune_similarity_model=fine_tune_similarity_model, document_db_file_path=document_db_file_path, vector_store_file_path=vector_store_file_path)
    return {"message": "Dialog agent reset successfully."}

@app.post("/perceive_text")
async def perceive_text(text_request: TextRequest):
    input_string = text_request.text 
    output_string = dialog_agent.generate_output(input_string)
    response = {"response": output_string}
    return response

@app.post("/update_table_state")
async def update_table_state(text_request: TextRequest):
    new_table_file_name = text_request.text 
    output_string = dialog_agent.update_table_state(new_table_file_name)
    response = {"response": output_string}
    return response

@app.get("/get_available_datasets")
async def get_available_datasets():
    # datasets are located in 'web_api/data/tables' -> get all file names in this directory
    dataset_names = os.listdir('./web_api/data/tables')
    # filter only .csv files
    dataset_names = [name for name in dataset_names if name.endswith('.csv')]
    # return as json
    response = {"response": {"dataset_names": dataset_names}}
    return response

@app.post("/add_online_training_sample")
async def add_online_training_sample(request: Request):
    online_training_sample = await request.json()
    dialog_agent.add_online_training_sample(online_training_sample)
    response = {"response": "OK"}
    return response

# ---------------------------------------------------------------------------------------------

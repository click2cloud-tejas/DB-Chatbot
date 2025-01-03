import os
import pandas as pd
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import pyodbc
import torch

# Path where FAISS index will be saved locally
FAISS_INDEX_PATH = "faiss_index_DB_5_dev5"

def fetch_data_from_sql(server, port, database, username, password, tables):
    conn = pyodbc.connect(
        f"DRIVER={{SQL Server}};SERVER={server},{port};DATABASE={database};UID={username};PWD={password}"
    )
    cursor = conn.cursor()

    # Execute the stored procedures
    cursor.execute("Exec SP_Bot_Test_All @ActionMode=1")
    row_all_1 = cursor.fetchall()
    columns_1 = [column[0] for column in cursor.description]

    # Execute the stored procedures
    cursor.execute("Exec SP_Bot_Test_All @ActionMode=2")
    row_all_2 = cursor.fetchall()
    columns_2 = [column[0] for column in cursor.description]

    # Execute the stored procedures
    cursor.execute("Exec SP_Bot_Test_All @ActionMode=3")
    row_all_3 = cursor.fetchall()
    columns_3 = [column[0] for column in cursor.description]

    # Execute the stored procedures
    cursor.execute("Exec SP_Bot_Test_All @ActionMode=6")
    row_all_6 = cursor.fetchall()
    columns_6 = [column[0] for column in cursor.description]

     # Execute the stored procedures
    cursor.execute("Exec SP_Bot_Test_All @ActionMode=7")
    row_all_7 = cursor.fetchall()
    columns_7 = [column[0] for column in cursor.description]

     # Execute the stored procedures
    cursor.execute("Exec SP_Bot_Test_All @ActionMode=13")
    row_all_13 = cursor.fetchall()
    columns_13 = [column[0] for column in cursor.description]



    documents = []

    if len(row_all_1) == 0:
        documents.append(Document(page_content="No Farm Details found for the given Farm Id and Farm Name"))
    else:
        for r in row_all_1:
            farm_info = (
                f"I am providing you with Farm Deatils about a farm named {r.FarmTitle}. "
                f"It is located in {r.FarmLocation}. The farm covers an area of {r.FarmArea} acres "
                f"and its coordinates are {r.FarmCoordinate}. Farm Id for this farm is {r.MasterFarmId}."
            )

        
            # print(farm_info)
            # Add both pieces of farm info as separate documents
            documents.append(Document(page_content=farm_info))
            



    if len(row_all_2) == 0:
        documents.append(Document(page_content="No Farm Details found for the given Farm Id and Farm Name"))
    else:
        for r in row_all_2:
            # print(row_all_2)
            # print(r)
            soil_info = (
                
                f"Maximum Value of {r.ParameterTitle} value is {r.MaxParameterValue} for the farm named  {r.FarmTitle}. "
                f"Latitude is {r.Latitude} and Longitude is {r.Longitude} for the Farm named {r.FarmTitle}"
            )

            # print(soil_info)
            # Add both pieces of farm info as separate documents
            documents.append(Document(page_content=soil_info))
            

    if len(row_all_3) == 0:
        documents.append(Document(page_content="No Farm Details found for the given Farm Id and Farm Name"))
    else:
        for r in row_all_3:
            climate_info = (
                f" The minimum temperature was : {r.MinTemp} on the the climate date {r.ClimateDate} for the farm named {r.FarmTitle} "
                f" The maximum temperature was : {r.MinTemp} on the the climate date {r.ClimateDate} for the farm named {r.FarmTitle}"
                f" The Humidity was {r.Humidity} on the the climate date {r.ClimateDate} for the farm named {r.FarmTitle}"
                f" The Wind Speed was : {r.Windspeed} on the the climate date {r.ClimateDate} for the farm named {r.FarmTitle} "

                
             
                
            )

            # print(climate_info)
            # Add both pieces of farm info as separate documents
            documents.append(Document(page_content=climate_info))
    

        

    if len(row_all_6) == 0:
        documents.append(Document(page_content="No Farm Details found for the given Farm Id and Farm Name"))
    else:
        for r in row_all_6:
            crop_plan_info = (
                # f"MasterFarmId of the Assessment named {r.AssessmentTitle} is {r.MasterFarmId} "
                f"Crop grown in the Assessment named {r.AssessmentTitle} is {r.CropTitle}. "
                
            )

        
            # print(crop_plan_info)
            # Add both pieces of farm info as separate documents
            documents.append(Document(page_content=crop_plan_info))

    if len(row_all_7) == 0:
        documents.append(Document(page_content="No Farm Details found for the given Farm Id and Farm Name"))
    else:
        for r in row_all_7:
            crop_stage_info = (
                # f"MasterFarmId of the Assessment named {r.AssessmentTitle} is {r.MasterFarmId} "
                #f"Crop grown in the Assessment named {r.AssessmentTitle} is {r.CropTitle}. "
                f"For the Assessment named {r.AssessmentTitle} for the crop {r.CropTitle}, Crop Operation is {r.OperationTitle} which starts from {r.StartDate} and ends on {r.EndDate} "
                f"For the Assessment named {r.AssessmentTitle} for the crop {r.CropTitle}, Crop stage is {r.CropOperationName} which starts from {r.StartDate} and ends on {r.EndDate} "
                
            )

        
            # print(crop_stage_info)
            # Add both pieces of farm info as separate documents
            documents.append(Document(page_content=crop_stage_info))


    if len(row_all_13) == 0:
        documents.append(Document(page_content="No Farm Details found for the given Farm Id and Farm Name"))
    else:
        for r in row_all_13:
            govt_scheme_info = (
                # f"MasterFarmId of the Assessment named {r.AssessmentTitle} is {r.MasterFarmId} "
                f"The Farm named {r.FarmTitle} is eligible with the government scheme named  {r.GovtSchemeTitle}  with the website link {r.GovtSchemeURL} "
                
            )


            # print(govt_scheme_info)
            # Add both pieces of farm info as separate documents
            documents.append(Document(page_content=govt_scheme_info))
                     

    conn.close()
    return documents

# Function to initialize the QA chain
def initialize_qa_chain(vectordb):
    CHECKPOINT = "MBZUAI/LaMini-T5-738M"
    TOKENIZER = AutoTokenizer.from_pretrained(CHECKPOINT)
    BASE_MODEL = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT, device_map=torch.device('cpu'), torch_dtype=torch.float32)
    pipe = pipeline(
        'text2text-generation',
        model=BASE_MODEL,
        tokenizer=TOKENIZER,
        max_length=1024,
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
        clean_up_tokenization_spaces=True,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        return_source_documents=True
    )
    return qa_chain

# Function to process the user's query
def process_answer(instruction, qa_chain):
    result = qa_chain.invoke({"query": instruction})
    source_docs = result.get('source_documents', [])
    if len(source_docs) == 0:
        return "Sorry, it is not provided in the given context."
    answer = result['result']
    return answer

# Function to load or create a vector database
def load_or_create_vectordb(embeddings, documents):
    if os.path.exists(FAISS_INDEX_PATH):
        print("Loading existing FAISS index...")
        vectordb = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Creating a new FAISS index...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=90)
        splits = text_splitter.split_documents(documents)
        vectordb = FAISS.from_documents(splits, embeddings)
        vectordb.save_local(FAISS_INDEX_PATH)
        print(f"New FAISS index created and saved to {FAISS_INDEX_PATH}.")
    
    return vectordb

def main():


    server = "20.174.45.61"
    port = "1433"
    database = "DBAgriPilot-dev"
    username = "sa"
    password = "NagpurAgri2050*"
    table_names = ['AMMasterFarm', 'AMMasterIndices']

    # Fetch data from the SQL tables
    documents = fetch_data_from_sql(server, port, database, username, password, table_names)
    if not documents:
        print("No data retrieved from the specified tables.")
        return

    # Generate embeddings and vector database
    embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
    vectordb = load_or_create_vectordb(embeddings, documents)

    # Initialize QA chain
    qa_chain = initialize_qa_chain(vectordb)
    print("Embeddings are ready. You can now ask questions about the data.")

    # Chat loop
    while True:
        prompt = input("\nYou: ")
        if prompt.lower() in ["exit", "quit"]:
            print("Exiting the chatbot.")
            break
        response = process_answer(prompt, qa_chain)
        print(f"Bot: {response}")


if __name__ == "__main__":
    main()




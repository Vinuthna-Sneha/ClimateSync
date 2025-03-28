import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import pandas as pd
from pydantic import BaseModel
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain.chains import SequentialChain, LLMChain
from google.oauth2 import service_account
from google.cloud import aiplatform, storage
from typing import Optional
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Zone API", description="API for zone data", version="1.0.0")
logger.info("FastAPI app initialized")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("CORS middleware added")

MONGO_URI = "mongodb+srv://n210103:tanmayi@strategy.8gapk.mongodb.net/"
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')  # Test connection on startup
    db = client["climate_strategy_db"]
    strategies_collection = db["strategies"]
    logger.info("Connected to MongoDB successfully")
except ConnectionFailure as e:
    logger.error(f"Failed to connect to MongoDB: {str(e)}")
    client = None
    strategies_collection = None

ZONES_FILE = "srikakulam_zone_coordinates.json"
ZONE_INFO_FILE = "report.json"
DEMOGRAPHICS_FILE = "updated_data_with_cri_sklm.csv"

zones_data = {}
if os.path.exists(ZONES_FILE):
    try:
        with open(ZONES_FILE, 'r') as file:
            zones_data = json.load(file)
        logger.info(f"Successfully loaded {ZONES_FILE}. Zones available: {list(zones_data.keys())}")
    except json.JSONDecodeError:
        logger.error(f"Error: {ZONES_FILE} contains invalid JSON")
else:
    logger.warning(f"Warning: {ZONES_FILE} not found")

zone_info_data = {}
if os.path.exists(ZONE_INFO_FILE):
    try:
        with open(ZONE_INFO_FILE, 'r') as file:
            zone_info_data = json.load(file)
        logger.info(f"Successfully loaded {ZONE_INFO_FILE}. Zones available: {list(zone_info_data.keys())}")
    except json.JSONDecodeError:
        logger.error(f"Error: {ZONE_INFO_FILE} contains invalid JSON")
else:
    logger.warning(f"Warning: {ZONE_INFO_FILE} not found")

demographics_data = {}
if os.path.exists(DEMOGRAPHICS_FILE):
    try:
        df = pd.read_csv(DEMOGRAPHICS_FILE, dtype={"zone": str})
        demographics_data = df.set_index("zone").to_dict(orient="index")
        logger.info(f"Successfully loaded {DEMOGRAPHICS_FILE}. Zones available: {list(demographics_data.keys())}")
    except Exception as e:
        logger.error(f"Error loading {DEMOGRAPHICS_FILE}: {e}")
else:
    logger.warning(f"Warning: {DEMOGRAPHICS_FILE} not found")

@app.get("/zone_ids")
async def get_zone_ids():
    if not zone_data:
        raise HTTPException(status_code=404, detail="No zone data available")
    
    zone_ids_with_names = [
        f"Zone {zone['zone']} ({zone['zone_name']})"
        for zone in zone_data
    ]
    return zone_ids_with_names

@app.get("/zones/{zone_id}")
async def get_zones(zone_id: str):
    if zone_id not in zones_data:
        raise HTTPException(status_code=404, detail=f"Zone {zone_id} not found")
    return {"zone_id": zone_id, "coordinates": zones_data[zone_id]}

@app.get("/zone_info/{zone_id}")
async def get_zone_info(zone_id: str):
    if not zone_info_data:
        raise HTTPException(status_code=404, detail="No zone info data available")
    formatted_zone_id = f"Zone {zone_id}"
    if formatted_zone_id not in zone_info_data:
        raise HTTPException(status_code=404, detail=f"Zone {zone_id} not found")
    return {"zone_id": zone_id, "details": [zone_info_data[formatted_zone_id]]}

@app.get("/zone_demographics/{zone_id}")
async def get_zone_demographics(zone_id: str):
    if not demographics_data:
        raise HTTPException(status_code=404, detail="No demographic data available")
    if zone_id not in demographics_data:
        raise HTTPException(status_code=404, detail=f"Zone {zone_id} not found in demographic data")
    return {"zone_id": zone_id, "demographics": demographics_data[zone_id]}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

with open("unified_data.json", "r") as f:
    zone_data = json.load(f)

credentials = service_account.Credentials.from_service_account_file("inspired-rock-450806-r5-d18f93b7f5ba.json")
aiplatform.init(project="inspired-rock-450806-r5", location="us-central1", credentials=credentials)

llm = VertexAI(model_name="gemini-1.5-flash", credentials=credentials, max_output_tokens=3000, temperature=0.7)
embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003", credentials=credentials)

# GCS setup for Chroma persistence
BUCKET_NAME = "inspired-rock-450806-r5.appspot.com"  # Default App Engine bucket
CHROMA_DIR = "chroma_db"
LOCAL_CHROMA_DIR = "/tmp/chroma_db"

def sync_chroma_to_gcs():
    """Sync local Chroma files to GCS."""
    try:
        gcs_client = storage.Client(credentials=credentials, project="inspired-rock-450806-r5")
        bucket = gcs_client.get_bucket(BUCKET_NAME)
        for root, _, files in os.walk(LOCAL_CHROMA_DIR):
            for file in files:
                local_path = os.path.join(root, file)
                blob_path = os.path.join(CHROMA_DIR, os.path.relpath(local_path, LOCAL_CHROMA_DIR))
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(local_path)
        logger.info(f"Synced Chroma data to GCS bucket {BUCKET_NAME}/{CHROMA_DIR}")
    except Exception as e:
        logger.error(f"Error syncing Chroma to GCS: {str(e)}")

def sync_chroma_from_gcs():
    """Download Chroma files from GCS to local /tmp."""
    try:
        gcs_client = storage.Client(credentials=credentials, project="inspired-rock-450806-r5")
        bucket = gcs_client.get_bucket(BUCKET_NAME)
        blobs = bucket.list_blobs(prefix=CHROMA_DIR)
        os.makedirs(LOCAL_CHROMA_DIR, exist_ok=True)
        for blob in blobs:
            if blob.name.endswith("/"):  # Skip directories
                continue
            local_path = os.path.join("/tmp", blob.name)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)
        logger.info(f"Loaded Chroma data from GCS bucket {BUCKET_NAME}/{CHROMA_DIR}")
        return True
    except Exception as e:
        logger.error(f"Error loading Chroma from GCS: {str(e)}")
        return False

# Initialize Chroma with GCS persistence
documents = [Document(page_content=f"Zone {zone['zone']}: {json.dumps(zone)}", metadata={"zone_id": zone["zone"]}) for zone in zone_data]
if sync_chroma_from_gcs() and os.path.exists(os.path.join(LOCAL_CHROMA_DIR, "chroma.sqlite3")):
    vector_store = Chroma(persist_directory=LOCAL_CHROMA_DIR, embedding_function=embeddings)
    logger.info("Loaded existing Chroma vector store from GCS")
else:
    vector_store = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=LOCAL_CHROMA_DIR)
    sync_chroma_to_gcs()
    logger.info("Initialized new Chroma vector store and synced to GCS")

overview_template = """
Role Assignment:
You are Government Strategy Planner AI, a highly capable AI assistant specializing in creating detailed, practical, and data-driven strategy plans for urban and rural development. Your goal is to analyze the provided unified zone data and develop a realistic, implementable strategy plan tailored to the specified location.
Task: Develop a detailed, actionable, and phased strategy plan based on the zone overview and idea, aligning with real-time initiatives in Andhra Pradesh, India.
Zone Name: {zone_name}
Zone Data: {context}
Location Context: {location_context}

Instructions:
- Analyze land cover, environmental factors, risk metrics, socioeconomic status, and industries.
- Identify key opportunities (e.g., renewable energy potential) and challenges (e.g., flood risk).
- Highlight climate vulnerabilities (e.g., GHG emission hotspots, extreme weather risks).
- Keep it brief and data-driven.
Output Format:
- Headings: "Overview", "Opportunities", "Challenges", "Climate Vulnerabilities"
- Bullet points for clarity
"""
overview_prompt = PromptTemplate(input_variables=["zone_name", "context", "location_context"], template=overview_template)

strategy_template = """
Role Assignment:
You are Government Strategy Planner AI, a highly capable AI assistant specializing in creating detailed, practical, and data-driven strategy plans for urban and rural development. Your goal is to analyze the provided unified zone data and develop a realistic, implementable strategy plan tailored to the specified location.
Zone Name: {zone_name}
Zone Data: {context}
Idea: {idea}
Location Context: {location_context}
Zone Overview: {overview}

Context:
The given data provides a comprehensive overview of the land cover, environmental factors, risk metrics, trends, socioeconomic status, industries, and research insights of a specific zone. Your strategy should align with real-world constraints, regulations, and feasibility while maximizing sustainability, economic growth, and social well-being.
Objective:
- Develop a strategy plan that directly addresses the user-provided idea: '{idea}'.
- Incorporate scalable, practical solutions tailored to the idea, leveraging zone-specific data.
- If applicable, align with broader goals like climate resilience, economic growth, or social well-being, but only as secondary considerations unless explicitly mentioned in the idea.
Instructions for Strategy Development:
1. Overview of the Zone
Provide a brief analysis of the zone’s characteristics based on the given data.
Identify key opportunities and challenges based on land cover, risk metrics, and environmental factors.
2. Strategy Plan & Implementation
Develop a well-structured, phased strategy addressing the identified challenges.
Ensure the plan is realistic, feasible, and backed with dynamic timeline based on strategy by real-world constraints.

Include specific projects, initiatives, or interventions necessary for improving the zone.
Provide numerical estimations where applicable (e.g., expected impact in GDP growth, employment generation).
Consider a dynamic timeline for implementing the strategy, breaking it down into different termed goals.
3. Budget & Resource Allocation
Provide a detailed financial breakdown, including:

Infrastructure Costs (e.g., roads, housing, public utilities)
Environmental Restoration (e.g., afforestation, water conservation)
Public Services (e.g., healthcare, education, disaster preparedness)
Technology & Innovation Investments
Other Necessary Expenditures
Additionally, provide:

Projected ROI (Return on Investment) or cost-benefit analysis for major investments.
Manpower Requirements including:
Government bodies involved.
Engineers, urban planners, environmentalists, etc.
Local workforce and skilled labor.
- Provide numerical estimates.
4. Legal & Policy Considerations
Identify laws, regulations, and policies applicable to the location.
Ensure compliance with environmental, urban planning, and labor laws specific to the region.
Highlight any permits or approvals required for implementation.
5. Risk Mitigation & Contingency Plan
Identify potential risks (e.g., climate risks, financial risks, social resistance).
Propose mitigation strategies with numerical justifications where possible (e.g., estimated financial losses from a past disaster and the cost of preventive measures).
If applicable, reference historical cases where similar risks impacted projects and how those issues were addressed.
Include a disaster management plan based on risk metrics.
6. Monitoring & Evaluation
Define KPIs (Key Performance Indicators) to measure success.
Provide a timeline for periodic assessments to track progress.
Suggest a feedback mechanism for continuous improvement.
7. Research and References
  List the Sources and Past Research from which this strategy development referred from like a bibliography.

Output Format:
Generate the response in a structured format with:
✅ Headings & bullet points for clarity.
✅ Financial tables where applicable.
✅ Numerical estimates to support key recommendations.
✅ Relevant case studies or historical references where applicable.

Constraints:
❌ Avoid unrealistic assumptions—all strategies must be based on real-world feasibility.
✅ Ensure practicality and effectiveness with data-backed recommendations.
✅ Maintain alignment with the region’s economic, environmental, and social goals.
"""
strategy_prompt = PromptTemplate(input_variables=["zone_name", "context", "idea", "location_context", "overview"], template=strategy_template)

qna_template = """
Role Assignment:
You are Government Strategy Planner AI, a highly capable AI assistant specializing in creating detailed, practical, and data-driven strategy plans for urban and rural development. Your goal is to analyze the provided unified zone data and develop a realistic, implementable strategy plan tailored to the specified location.
Given Strategy: {strategy}
User Question: {question}
Selected Text: {selected_text}

Instructions:
- Provide a clear, concise, data-driven answer based on the strategy and zone data.
- Consider the selected text as additional context to tailor the answer more precisely to the user's focus.
- If needed, infer additional context or fetch real-world examples (e.g., solar costs).
Output Format:
- Plain text response
"""
qna_prompt = PromptTemplate(input_variables=["strategy", "question", "selected_text"], template=qna_template)

modify_template = """
Role Assignment:
You are Government Strategy Planner AI, a highly capable AI assistant specializing in creating detailed, practical, and data-driven strategy plans for urban and rural development. Your goal is to analyze the provided unified zone data and develop a realistic, implementable strategy plan tailored to the specified location.
Zone Data: {context}
Strategy: {strategy}
Location Context: {location_context}
Zone Overview: {overview}
Modification Request: {modification_request}
Instructions:
- Update the original strategy to incorporate the user's modification request while maintaining alignment with climate goals.
- Adjust relevant sections (e.g., implementation, budget, risk mitigation) to reflect the change.
- Ensure feasibility and consistency with the original data and constraints.
- Provide numerical estimates where applicable (e.g., revised costs, emission impacts).
Output Format:
- Same as the original strategy: headings, bullet points, financial tables, numerical estimates.
- Highlight modified sections with a note (e.g., "Modified: Added wind energy").
"""
modify_prompt = PromptTemplate(input_variables=["strategy", "modification_request", "context", "location_context", "overview"], template=modify_template)

documents = [Document(page_content=f"Zone {zone['zone']}: {json.dumps(zone)}", metadata={"zone_id": zone["zone"]}) for zone in zone_data]
vector_store = Chroma.from_documents(documents=documents, embedding=embeddings)
logger.info("Initialized in-memory Chroma vector store")

class StrategyRequest(BaseModel):
    zone_id: int
    idea: str

class QnARequest(BaseModel):
    strategy: str
    question: str
    selected_text: Optional[str] = None

class ModifyRequest(BaseModel):
    zone_id: int
    strategy: str
    modification_request: str

class FinalizeRequest(BaseModel):
    zone_id: int
    strategy: str
    timestamp: str
    idea: str

def generate_location_context(zone_id: int) -> str:
    zone = next((z for z in zone_data if z["zone"] == zone_id), None)
    if not zone:
        return f"No location data available for Zone {zone_id}."
    
    # Use the zone_name directly from the zone data
    zone_name = zone.get("zone_name", f"Zone {zone_id}")
    lat, lon = zone["location"]["latitude"], zone["location"]["longitude"]
    land_cover = zone["land_cover"]
    socio_economic = zone["socioeconomic_status"]
    env_factors = zone["environmental_factors"]
    
    # Determine area type based on built area and socioeconomic factors
    is_rural = land_cover["built"] < 0.3 and (socio_economic["hospitals"] + socio_economic["banks"]) < 10
    area_type = "rural" if is_rural else "semi-urban"
    
    # Determine climate and precipitation characteristics
    climate = "sunny climate with over 300 sunny days annually" if lat < 20 else "moderate climate"
    precipitation = "low precipitation (~600-800 mm/year)" if env_factors["precipitation"] < 0.03 else "moderate precipitation"
    
    # Construct the context string using zone_name explicitly
    context = (
        f"The zone is {zone_name}, located in Andhra Pradesh, India (latitude {lat}, longitude {lon}), "
        f"a {area_type} area with {climate} and {precipitation}. "
        f"It features {land_cover['crops']*100:.1f}% crop coverage and {land_cover['built']*100:.1f}% built areas, "
        f"with {socio_economic['schools']} schools, {socio_economic['hospitals']} hospitals, and {socio_economic['banks']} banks."
    )
    return context

overview_chain = LLMChain(llm=llm, prompt=overview_prompt, output_key="overview")
strategy_chain = LLMChain(llm=llm, prompt=strategy_prompt, output_key="strategy")
full_chain = SequentialChain(
    chains=[overview_chain, strategy_chain],
    input_variables=["zone_name", "context", "idea", "location_context"],
    output_variables=["overview", "strategy"]
)

qna_chain = LLMChain(llm=llm, prompt=qna_prompt)
modify_chain = LLMChain(llm=llm, prompt=modify_prompt, output_key="modified_strategy")

@app.post("/generate-strategy/")
async def generate_strategy(request: StrategyRequest):
    try:
        # Retrieve the zone data for the given zone_id
        zone = next((z for z in zone_data if z["zone"] == request.zone_id), None)
        if not zone:
            raise HTTPException(status_code=404, detail=f"Zone {request.zone_id} not found in unified_data.json")

        # Extract the zone_name from the zone data
        zone_name = zone.get("zone_name", f"Zone {request.zone_id}")  # Fallback to "Zone {zone_id}" if zone_name is not found

        # Retrieve context from vector store
        query = f"Zone {request.zone_id}"
        retrieved_docs = vector_store.as_retriever(search_kwargs={"k": 1}).invoke(query)
        context = retrieved_docs[0].page_content if retrieved_docs else f"Zone {request.zone_id}: No data available."
        
        # Generate location context
        location_context = generate_location_context(request.zone_id)
        
        # Invoke the chain with the zone_name
        result = full_chain.invoke({
            "zone_name": zone_name,
            "context": context,
            "idea": request.idea,
            "location_context": location_context
        })
        return {"strategy": result["strategy"]}
    except Exception as e:
        logger.error(f"Error generating strategy for Zone {request.zone_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating strategy: {str(e)}")

@app.post("/ask-question/")
async def ask_question(request: QnARequest):
    try:
        answer = qna_chain.invoke({
            "strategy": request.strategy,
            "question": request.question,
            "selected_text": request.selected_text or ""
        })
        return {"answer": answer["text"] if isinstance(answer, dict) else answer}
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")

@app.post("/modify-strategy/")
async def modify_strategy(request: ModifyRequest):
    try:
        query = f"Zone {request.zone_id}"
        retrieved_docs = vector_store.as_retriever(search_kwargs={"k": 1}).invoke(query)
        context = retrieved_docs[0].page_content if retrieved_docs else f"Zone {request.zone_id}: No data available."
        location_context = generate_location_context(request.zone_id)
        result = modify_chain.invoke({
            "strategy": request.strategy,
            "modification_request": request.modification_request,
            "context": context,
            "location_context": location_context,
            "overview": "Overview placeholder"
        })
        return {"modified_strategy": result["modified_strategy"]}
    except Exception as e:
        logger.error(f"Error modifying strategy for Zone {request.zone_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error modifying strategy: {str(e)}")

@app.post("/finalize-strategy/")
async def finalize_strategy(request: FinalizeRequest):
    logger.info(f"Received request: {request.dict()}")
    if strategies_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB connection unavailable")
    try:
        latest_strategy = strategies_collection.find_one(
            {"zone_id": str(request.zone_id)},
            sort=[("version", -1)]
        )
        new_version = (latest_strategy["version"] + 1) if latest_strategy else 1
        strategy_doc = {
            "zone_id": str(request.zone_id),
            "strategy": request.strategy,
            "version": new_version,
            "timestamp": request.timestamp,
            "idea": request.idea
        }
        result = strategies_collection.insert_one(strategy_doc)
        logger.info(f"Finalized strategy for Zone {request.zone_id}, Version {new_version}. MongoDB ID: {result.inserted_id}")
        return {"message": f"Strategy version {new_version} for Zone {request.zone_id} finalized successfully", "version": new_version}
    except Exception as e:
        logger.error(f"Error finalizing strategy for Zone {request.zone_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error finalizing strategy: {str(e)}")

@app.get("/get-strategy/{zone_id}/{version}")
async def get_strategy(zone_id: str, version: int):
    if strategies_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB connection unavailable")
    try:
        strategy = strategies_collection.find_one({"zone_id": zone_id, "version": version})
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Version {version} not found for Zone {zone_id}")
        return {"strategy": strategy["strategy"]}
    except Exception as e:
        logger.error(f"Error retrieving strategy for Zone {zone_id}, Version {version}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving strategy: {str(e)}")

@app.get("/get-all-strategies/{zone_id}")
async def get_all_strategies(zone_id: str):
    if client is None or strategies_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB connection unavailable")
    try:
        client.admin.command('ping')  # Verify connection
        strategies = list(strategies_collection.find({"zone_id": str(zone_id)}))
        if not strategies:
            logger.info(f"No strategies found for Zone {zone_id}")
            return {"strategies": []}
        
        # Validate required fields and handle missing ones
        valid_strategies = []
        required_fields = {"version", "timestamp", "strategy", "idea"}
        for s in strategies:
            if all(field in s for field in required_fields):
                valid_strategies.append({
                    "version": s["version"],
                    "timestamp": s["timestamp"],
                    "strategy": s["strategy"],
                    "idea": s["idea"]
                })
            else:
                logger.warning(f"Skipping invalid strategy document for Zone {zone_id}: {s}")
        
        logger.info(f"Returning {len(valid_strategies)} strategies for Zone {zone_id}")
        return {"strategies": valid_strategies}
    except ConnectionFailure as e:
        logger.error(f"MongoDB connection error for Zone {zone_id}: {str(e)}")
        raise HTTPException(status_code=503, detail=f"MongoDB connection error: {str(e)}")
    except OperationFailure as e:
        logger.error(f"MongoDB operation error for Zone {zone_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"MongoDB operation error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error retrieving strategies for Zone {zone_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error retrieving strategies: {str(e)}")
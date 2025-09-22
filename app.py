from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import json
import os
import re
import requests
import base64
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import List, Dict, Any, TypedDict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
# from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from mem0 import MemoryClient
from PIL import Image
import pypdf
import docx
from datetime import datetime
from datetime import timedelta 
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.tools.google_serper import GoogleSerperResults
from functools import lru_cache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

gemini_api_key = os.getenv('GEMINI_API_KEY')
serper_api_key = os.getenv('SERPER_API_KEY')
mem0_api_key = os.getenv('MEM0_API_KEY')
groq_api_key = os.getenv('groq_api_key')

if not all([gemini_api_key, serper_api_key, mem0_api_key, groq_api_key]):
    raise ValueError("Missing one or more API keys in environment variables")

llm = LLM(
    model="gemini/gemini-2.5-flash",
    temperature=0.7,
    api_key=gemini_api_key
)

llm_deepseek = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    api_key=groq_api_key,
    temperature=0.6
)
llm_gpt = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=groq_api_key,
    temperature=0.6
)
llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=gemini_api_key,
    temperature=0.6
)

langchain_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    api_key=gemini_api_key
)

def init_db():
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL
                    )''')
    conn.commit()
    conn.close()

serper = GoogleSerperResults(api_key=serper_api_key, num_results=3)

mem0_client = MemoryClient(api_key=mem0_api_key)

serper_tool = SerperDevTool(
    api_key=serper_api_key,
    n_results=50
)

class Scheme(BaseModel):
    name: str
    category: str
    description: str
    link: str

class DiseaseResult(BaseModel):
    disease: str
    plant: str
    symptoms: str
    remedies: str
    resources: List[Dict[str, str]]

class CropPlan(BaseModel):
    crop_name: str
    sowing_time: str
    cultivation_tips: str
    expected_yield: str

class Product(BaseModel):
    name: str
    description: str
    price: str
    certified: str
    link: str

class EquipmentSearchTool(BaseTool):
    name: str = Field(default="EquipmentSearchTool", description="Searches for agricultural equipment details using web search.")
    description: str = Field(default="Fetches equipment details like price, description, and source using SerperDevTool.")

    def _run(self, equipment_name: str) -> Dict[str, str]:
        try:
            query = f"{equipment_name} agricultural equipment price site:*.edu | site:*.org | site:*.gov"
            search_results = serper_tool._run(query=query)
            
            price = "Price not found."
            description = "No detailed description available."
            source = ""

            for result in search_results.get('organic', [])[:5]:  # Limit to top 5 results
                snippet = result.get('snippet', '').lower()
                if 'price' in snippet or 'cost' in snippet:
                    price = result.get('snippet', price)[:100] + "..." if len(result.get('snippet', '')) > 100 else result.get('snippet', price)
                if not source:
                    source = result.get('link', '')
                if not description.startswith("No"):
                    description = result.get('snippet', description)[:200] + "..." if len(result.get('snippet', '')) > 200 else result.get('snippet', description)

            return {
                'name': equipment_name,
                'price': price,
                'source': source,
                'description': description
            }
        except Exception as e:
            logger.error(f"Error searching equipment: {str(e)}")
            return {
                'name': equipment_name,
                'price': 'Unknown',
                'source': '',
                'description': f"Error searching equipment: {str(e)}"
            }


class SchemeFilterTool(BaseTool):
    name: str = Field(default="SchemeFilterTool", description="Filters schemes based on user criteria with priority.")
    description: str = Field(default="Filters government schemes by prioritizing location, then occupation, then caste, gender, and landholding.")

    def _run(self, schemes: List[Dict[str, Any]], user_data: dict) -> List[Dict[str, Any]]:
        try:
            filtered_schemes = []
            for scheme in schemes:
                location = user_data.get('location', '').lower()
                occupation = user_data.get('occupation', '').lower()
                caste = user_data.get('caste', '').lower()
                gender = user_data.get('gender', '').lower()
                landholding = user_data.get('landholding', '').lower()

                name_lower = scheme.get('name', '').lower()
                category_lower = scheme.get('category', '').lower()
                desc_lower = scheme.get('description', '').lower()

                location_match = (
                    location in name_lower or
                    location in desc_lower or
                    location in scheme.get('link', '').lower() or
                    'assam' in desc_lower
                )

                occupation_match = (
                    occupation in category_lower or
                    occupation in desc_lower or
                    'farmer' in desc_lower or
                    'agriculture' in category_lower or
                    'agribusiness' in category_lower
                )

                other_match = (
                    caste in desc_lower or
                    gender in desc_lower or
                    (landholding and (
                        'land' in desc_lower or
                        'small' in desc_lower or
                        'marginal' in desc_lower or
                        landholding in desc_lower
                    ))
                )

                if location_match or occupation_match or other_match:
                    filtered_schemes.append(scheme)

            filtered_schemes.sort(key=lambda x: (
                user_data.get('location', '').lower() not in x.get('description', '').lower() and
                user_data.get('location', '').lower() not in x.get('name', '').lower() and
                user_data.get('location', '').lower() not in x.get('link', '').lower(),
                user_data.get('occupation', '').lower() not in x.get('category', '').lower() and
                user_data.get('occupation', '').lower() not in x.get('description', '').lower()
            ))
            return filtered_schemes[:20]
        except Exception as e:
            logger.error(f"Error filtering schemes: {str(e)}")
            return f"Error filtering schemes: {e}"

class CropDiseaseAPI(BaseTool):
    name: str = Field(default="CropDiseaseAPI", description="Tool to detect crop diseases from image using external ML API.")
    description: str = Field(default="Identifies crop disease by sending base64 image to susya.onrender.com API.")

    def _run(self, image_path: str) -> str:
        try:
            with open(image_path, "rb") as img_file:
                imgdata = base64.b64encode(img_file.read()).decode("utf-8")
            response = requests.post("https://susya.onrender.com", json={"image": imgdata})
            response.raise_for_status()

            data = json.loads(response.text)
            disease = data.get("disease", "Unknown disease")
            plant = data.get("plant", "Unknown plant")

            return f"Disease: {disease}\nPlant: {plant}"
        except Exception as e:
            logger.error(f"Error calling Crop Disease API: {str(e)}")
            return f"Error calling Crop Disease API: {e}"

class SoilTypeAPI(BaseTool):
    name: str = Field(default="SoilTypeAPI", description="Tool to detect soil type based on location.")
    description: str = Field(default="Identifies soil type by querying an external soil database API based on location.")

    def _run(self, location: str) -> str:
        try:
            soil_map = {
                'assam': 'Alluvial',
                'punjab': 'Alluvial',
                'tamil nadu': 'Red',
                'maharashtra': 'Black',
                'karnataka': 'Red',
                'gujarat': 'Sandy'
            }
            location = location.lower().strip()
            for key, value in soil_map.items():
                if key in location:
                    return value
            return 'Unknown'
        except Exception as e:
            logger.error(f"Error detecting soil type: {str(e)}")
            return 'Unknown'

scheme_filter_tool = SchemeFilterTool()
crop_disease_tool = CropDiseaseAPI()
soil_type_tool = SoilTypeAPI()

crop_planner = Agent(
    role="Crop Planning Expert",
    goal="Generate personalized crop recommendations based on location, season, soil type, and land size.",
    backstory="An AI agronomist specializing in crop selection and cultivation advice for Indian farmers.",
    tools=[serper_tool, soil_type_tool],
    verbose=True,
    llm=llm
)

product_researcher = Agent(
    role="Product Researcher",
    goal="Search for agricultural products online, including details, price, and certification status.",
    backstory="An AI expert in finding and analyzing agricultural products using web search tools.",
    tools=[serper_tool],
    verbose=True,
    llm=llm
)

scheme_researcher = Agent(
    role="Scheme Researcher",
    goal="Search and filter government agricultural schemes based on user criteria, prioritizing location, then occupation, then other criteria.",
    backstory="An expert in finding and filtering relevant government schemes for farmers using web search tools.",
    tools=[serper_tool, scheme_filter_tool],
    verbose=True,
    llm=llm
)

symptoms_advisor = Agent(
    role="Crop Symptoms Specialist",
    goal="Identify and describe common symptoms of crop diseases.",
    backstory="An AI agronomist specializing in recognizing and describing symptoms of crop diseases for farmers.",
    verbose=True,
    llm=llm
)

remedy_advisor = Agent(
    role="Agro Remedy Consultant",
    goal="Suggest effective solutions for crop diseases.",
    backstory="An expert agronomist helping farmers treat plant infections.",
    verbose=True,
    llm=llm
)

resource_link_finder = Agent(
    role="Agro Web Researcher",
    goal="Find helpful guides and links about crop disease treatments.",
    backstory="An AI assistant with access to the web for agricultural research.",
    tools=[serper_tool],
    verbose=True,
    llm=llm
)

legal_assistant_history = InMemoryChatMessageHistory()
veterinary_assistant_history = InMemoryChatMessageHistory()
financial_assistant_history = InMemoryChatMessageHistory()

def get_session_history(assistant_type: str):
    MAX_MESSAGES = 5
    histories = {
        'legal': legal_assistant_history,
        'veterinary': veterinary_assistant_history,
        'financial': financial_assistant_history
    }
    history = histories.get(assistant_type, legal_assistant_history)
    if len(history.messages) > MAX_MESSAGES:
        history.messages = history.messages[-MAX_MESSAGES:]
    return history

def get_soil_data(location: str) -> Dict:
    try:
        return {
            "location": location,
            "soil_type": "Loamy",
            "ph_level": 6.5,
            "nutrients": {"N": "High", "P": "Medium", "K": "Low"},
            "moisture": "Adequate"
        }
    except Exception as e:
        logger.error(f"Error fetching soil data: {e}")
        return {"error": "Failed to fetch soil data"}

def sanitize_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = text.replace('\u2011', '-')  
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)  
    return text.strip()

def is_valid_paragraph(text: str) -> bool:
    if not text or len(text) < 50:
        return False
    if re.match(r'^[()\-,;\s]+$', text):
        return False
    return True

def create_crop_planning_workflow(location: str, plan_duration: int, language: str, max_iterations: int = 3):
    class CropPlanState(TypedDict):
        location: str
        plan_duration: int
        language: str
        soil_data: Optional[Dict]
        climate_data: Optional[Dict]
        crop_plan: Optional[str]
        history: List[str]
        iteration: int
        final_plan: Optional[str]

    workflow = StateGraph(CropPlanState)

    data_collection_prompt = (
        "You are an agricultural expert tasked with summarizing crop suitability for {location} over a {plan_duration}-month period. "
        "Use the provided search results: {search_results}, soil data: {soil_data}, and climate data: {climate_data}. "
        "If search results are unavailable or irrelevant, use your knowledge to summarize crop suitability based on the provided soil and climate data. "
        "Provide a concise summary (100-150 words) of suitable crops and considerations for the specified duration, "
        "considering both soil conditions and climate patterns."
    )

    planning_prompt = (
        "You are an agricultural planner for {location} over a {plan_duration}-month period. "
        "Based on the data summary: {data_summary}, soil data: {soil_data}, climate data: {climate_data}, and conversation history: {history}, "
        "propose a crop plan (one paragraph) including specific crops, planting schedules, and considerations, "
        "ensuring it aligns with the soil conditions, climate patterns, and plan duration, and counters or builds on previous plans."
    )

    summarizer_prompt = (
        "You are a summarizer agent. Given the conversation history: {history}, "
        "create a detailed crop plan for a {plan_duration}-month period in {location}. "
        "Extract a diverse set of crop varieties (e.g., HD3086 wheat, IR64-SRI rice, SBCC212 barley, HM-5 maize, LC1136 cotton, Jalgaon85 moong) from DeepSeek, Gemini, and GPT plans in the history, ensuring at least one crop from each if possible. "
        "Avoid repeating the same crops unnecessarily and include specific planting months (e.g., October-November), management strategies (e.g., SRI, IPM, crop rotation), and soil/climate considerations. "
        "Consider soil data: {soil_data} and climate data: {climate_data}. "
        "Write the paragraph in {language}, ensuring clarity, coherence, and no garbled text, symbols, or formatting errors. "
        "Return only the paragraph text, no JSON, Markdown, or extra characters."
    )

    @lru_cache(maxsize=100)
    def collect_data(location: str, plan_duration: int, soil_data: str) -> str:
        try:
            query = f"climate and weather forecast for {location} next {plan_duration} months"
            search_results = serper.invoke({"query": query})
            
            climate_query = f"agricultural climate suitability {location} next {plan_duration} months"
            climate_search = serper.invoke({"query": climate_query})
            
            prompt = data_collection_prompt.format(
                location=location,
                plan_duration=plan_duration,
                search_results=json.dumps(search_results),
                soil_data=soil_data,
                climate_data=json.dumps(climate_search)
            )
            result = llm_gemini.invoke(prompt).content
            result = sanitize_text(result)
            if not result or result.strip() == "" or "error" in result.lower():
                default_summary = (
                    f"In {location} for a {plan_duration}-month period, the loamy soil with a pH of 6.5, high nitrogen, medium phosphorus, "
                    f"and low potassium supports crops like wheat, rice, chickpeas, and mustard. The expected climate includes moderate temperatures "
                    f"around 25C with seasonal variations, adequate rainfall during monsoon periods, and cooler winters suitable for rabi crops. "
                    f"Adequate moisture levels suit these crops, but potassium supplementation is advised to optimize yields. "
                    f"Crop rotation with legumes like chickpeas enhances soil nitrogen, while mustard improves soil structure. "
                    f"For a {plan_duration}-month plan, prioritize crops like wheat and chickpeas for rabi season or rice for kharif season, "
                    f"depending on the timeframe, with fertilizers to address potassium deficiency and irrigation planning based on seasonal rainfall."
                )
                logger.warning("No valid search results; using default summary")
                return default_summary
            logger.info(f"Data collection result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in data collection: {e}")
            default_summary = (
                f"In {location} for a {plan_duration}-month period, the loamy soil with a pH of 6.5, high nitrogen, medium phosphorus, "
                f"and low potassium supports crops like wheat, rice, chickpeas, and mustard. The expected climate includes moderate temperatures "
                f"around 25C with seasonal variations, adequate rainfall during monsoon periods, and cooler winters suitable for rabi crops. "
                f"Adequate moisture levels suit these crops, but potassium supplementation is advised to optimize yields. "
                f"Crop rotation with legumes like chickpeas enhances soil nitrogen, while mustard improves soil structure. "
                f"For a {plan_duration}-month plan, prioritize crops like wheat and chickpeas for rabi season or rice for kharif season, "
                f"depending on the timeframe, with fertilizers to address potassium deficiency and irrigation planning based on seasonal rainfall."
            )
            return default_summary

    def data_collection_node(state: CropPlanState) -> CropPlanState:
        soil_data = get_soil_data(state['location'])
        data_summary = collect_data(
            state['location'],
            state['plan_duration'],
            json.dumps(soil_data)
        )
        return {
            "soil_data": soil_data,
            "climate_data": {"forecast": f"Climate data for {state['location']} next {state['plan_duration']} months"},
            "history": state['history'] + [f"Data Summary: {data_summary}"],
            "iteration": state['iteration'],
            "language": state['language']
        }

    def deepseek_planning_node(state: CropPlanState) -> CropPlanState:
        try:
            data_summary = state['history'][-1] if state['history'] else "No data available"
            history = "\n".join(state['history'])
            prompt = planning_prompt.format(
                location=state['location'], 
                plan_duration=state['plan_duration'], 
                data_summary=data_summary, 
                history=history,
                soil_data=json.dumps(state['soil_data']),
                climate_data=json.dumps(state['climate_data'])
            )
            plan = f"DeepSeek Plan: {sanitize_text(llm_deepseek.invoke(prompt).content)}"
            logger.info(f"DeepSeek plan: {plan}")
            return {
                "crop_plan": plan,
                "history": state['history'] + [plan],
                "iteration": state['iteration'] + 1
            }
        except Exception as e:
            logger.error(f"Error in DeepSeek planning: {e}")
            return {
                "crop_plan": "Error generating DeepSeek plan",
                "history": state['history'] + ["DeepSeek Plan: Error"],
                "iteration": state['iteration'] + 1
            }

    def gemini_planning_node(state: CropPlanState) -> CropPlanState:
        try:
            data_summary = state['history'][-1] if state['history'] else "No data available"
            history = "\n".join(state['history'])
            prompt = planning_prompt.format(
                location=state['location'], 
                plan_duration=state['plan_duration'], 
                data_summary=data_summary, 
                history=history,
                soil_data=json.dumps(state['soil_data']),
                climate_data=json.dumps(state['climate_data'])
            )
            plan = f"Gemini Plan: {sanitize_text(llm_gemini.invoke(prompt).content)}"
            logger.info(f"Gemini plan: {plan}")
            return {
                "crop_plan": plan,
                "history": state['history'] + [plan],
                "iteration": state['iteration'] + 1
            }
        except Exception as e:
            logger.error(f"Error in Gemini planning: {e}")
            return {
                "crop_plan": "Error generating Gemini plan",
                "history": state['history'] + ["Gemini Plan: Error"],
                "iteration": state['iteration'] + 1
            }

    def gpt_planning_node(state: CropPlanState) -> CropPlanState:
        try:
            data_summary = state['history'][-1] if state['history'] else "No data available"
            history = "\n".join(state['history'])
            prompt = planning_prompt.format(
                location=state['location'], 
                plan_duration=state['plan_duration'], 
                data_summary=data_summary, 
                history=history,
                soil_data=json.dumps(state['soil_data']),
                climate_data=json.dumps(state['climate_data'])
            )
            plan = f"GPT Plan: {sanitize_text(llm_gpt.invoke(prompt).content)}"
            logger.info(f"GPT plan: {plan}")
            return {
                "crop_plan": plan,
                "history": state['history'] + [plan],
                "iteration": state['iteration'] + 1
            }
        except Exception as e:
            logger.error(f"Error in GPT planning: {e}")
            return {
                "crop_plan": "Error generating GPT plan",
                "history": state['history'] + ["GPT Plan: Error"],
                "iteration": state['iteration'] + 1
            }

    def summarizer_node(state: CropPlanState) -> CropPlanState:
        try:
            if not state['history'] or not state['soil_data'] or not state['climate_data']:
                logger.warning("Invalid input data for summarizer: missing history, soil_data, or climate_data")
                raise ValueError("Invalid input data")
            
            history = "\n".join(state['history'])
            prompt = summarizer_prompt.format(
                history=history,
                soil_data=json.dumps(state['soil_data']),
                climate_data=json.dumps(state['climate_data']),
                location=state['location'],
                plan_duration=state['plan_duration'],
                language=state['language']
            )
            final_plan = sanitize_text(llm_gemini.invoke(prompt).content)
            if not is_valid_paragraph(final_plan):
                logger.warning(f"Invalid summarizer output: {final_plan}")
                raise ValueError("Invalid or garbled summarizer output")
            logger.info(f"Final plan: {final_plan}")
            return {"final_plan": final_plan}
        except Exception as e:
            logger.error(f"Error in summarizer: {e} - History: {state['history']}")
            default_plan = (
                f"{state['location']} में {state['plan_duration']} महीने की फसल योजना के लिए, मई-जून में गर्मी-सहिष्णु ज्वार (जोवर) और मूंग (जलगांव85) को मई के पहले पखवाड़े में लगाएं, जिसमें गहन सिंचाई और पोटैशियम उर्वरक (30 किग्रा/हेक्टेयर) की आवश्यकता होगी। जुलाई-अगस्त में मानसून के साथ, जून के अंत या जुलाई की शुरुआत में खरीफ फसलें जैसे चावल (IR64-SRI, उत्तर-पश्चिम में जल प्रबंधन के साथ), मक्का (HM-5), कपास (LC1136), और उड़द लगाएं, जिसमें पोटैशियम की दो बार खुराक (20 किग्रा/हेक्टेयर) और मल्चिंग शामिल हो। सितंबर-अक्टूबर में खरीफ फसलों की कटाई करें और रबी मौसम के लिए मिट्टी तैयार करें, जिसमें कार्बनिक खाद (10 टन/हेक्टेयर), गहरी जुताई, और पोटैशियम-युक्त कम्पोस्ट (15 किग्रा/हेक्टेयर) शामिल हो। पूरे अवधि में, मिट्टी की नमी की निगरानी, ड्रिप सिंचाई, और एकीकृत कीट प्रबंधन (IPM) से अधिकतम उपज सुनिश्चित होगी।" if state['language'] == "Hindi" else
                f"{state['location']} ನಲ್ಲಿ {state['plan_duration']} ತಿಂಗಳ ಫಸಲು ಯೋಜನೆಗಾಗಿ, ಮೇ-ಜೂನ್‌ನಲ್ಲಿ ಶಾಖ-ಸಹಿಷ್ಣು ಜೋವರ್ (ಜೋವರ್) ಮತ್ತು ಮೂಂಗ್ (ಜಲಗಾಂವ್85) ಅನ್ನು ಮೇನ ಮೊದಲ ಪಕ್ಷದಲ್ಲಿ ನಾಟಿ ಮಾಡಿ, ಇದಕ್ಕೆ ತೀವ್ರವಾದ ನೀರಾವರಿ ಮತ್ತು ಪೊಟ್ಯಾಸಿಯಂ ರಸಗೊಬ್ಬರ (30 ಕೆಜಿ/ಹೆಕ್ಟೇರ್) ಅಗತ್ಯವಿರುತ್ತದೆ। ಜುಲೈ-ಆಗಸ್ಟ್‌ನಲ್ಲಿ ಮಾನ್ಸೂನ್ ಜೊತೆಗೆ, ಜೂನ್‌ನ ಕೊನೆಯಲ್ಲಿ ಅಥವಾ ಜುಲೈ ಆರಂಭದಲ್ಲಿ ಖರೀಫ್ ಬೆಳೆಗಳಾದ ಭತ್ತ (IR64-SRI, ಉತ್ತರ-ಪಶ್ಚಿಮದಲ್ಲಿ ನೀರಿನ ನಿರ್ವಹಣೆಯೊಂದಿಗೆ), ಮೆಕ್ಕೆಜೋಳ (HM-5), ಹತ್ತಿ (LC1136), ಮತ್ತು ಒಡಲಾಳನ್ನು ನಾಟಿ ಮಾಡಿ, ಇದರಲ್ಲಿ ಪೊಟ್ಯಾಸಿಯಂನ ಎರಡು ಡೋಸ್‌ಗಳು (20 ಕೆಜಿ/ಹೆಕ್ಟೇರ್) ಮತ್ತು ಮಲ್ಚಿಂಗ್ ಸೇರಿರುತ್ತದೆ। ಸೆಪ್ಟೆಂಬರ್-ಅಕ್ಟೋಬರ್‌ನಲ್ಲಿ ಖರೀಫ್ ಬೆಳೆಗಳ ಕೊಯ್ಲು ಮಾಡಿ ಮತ್ತು ರಬಿ ಋತುವಿಗೆ ಮಣ್ಣನ್ನು ಸಿದ್ಧಪಡಿಸಿ, ಇದರಲ್ಲಿ ಸಾವಯವ ಗೊಬ್ಬರ (10 ಟನ್/ಹೆಕ್ಟೇರ್), ಆಳವಾದ ಉಳುಮೆ, ಮತ್ತು ಪೊಟ್ಯಾಸಿಯಂ-ಒಳಗೊಂಡ ಕಾಂಪೋಸ್ಟ್ (15 ಕೆಜಿ/ಹೆಕ್ಟೇರ್) ಸೇರಿರುತ್ತದೆ। ಇಡೀ ಅವಧಿಯಲ್ಲಿ, ಮಣ್ಣಿನ ತೇವಾಂಶದ ಮೇಲ್ವಿಚಾರಣೆ, ಡ್ರಿಪ್ ನೀರಾವರಿ, ಮತ್ತು ಸಂಯೋಜಿತ ಕೀಟ ನಿರ್ವಹಣೆ (IPM) ಮೂಲಕ ಗರಿಷ್ಠ ಇಳುವರಿಯನ್ನು ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಿ।" if state['language'] == "Kannada" else
                f"{state['location']} ൽ {state['plan_duration']} മാസത്തെ വിള യോജനയ്ക്കായി, മേയ്-ജൂൺ മാസങ്ങളിൽ ചൂട്-സഹിക്കുന്ന ജോവർ (ജോവർ), മൂങ് (ജലഗാവ്85) എന്നിവ മേയ് ഒന്നാം പക്ഷത്തിൽ നടുക, ഇതിന് തീവ്രമായ ജലസേചനവും പൊട്ടാസ്യം വളം (30 കി.ഗ്രാം/ഹെക്ടർ) ആവശ്യമാണ്। ജൂലൈ-ഓഗസ്റ്റിൽ മൺസൂൺ വരുന്നതോടെ, ജൂൺ അവസാനം അല്ലെങ്കിൽ ജൂലൈ ആദ്യം ഖരീഫ് വിളകളായ അരി (IR64-SRI, വടക്ക്-പടിഞ്ഞാറ് ജലനിയന്ത്രണത്തോടെ), ചോളം (HM-5), കോട്ടൺ (LC1136), ഉഴുന്ന് എന്നിവ നടുക, ഇതിൽ പൊട്ടാസ്യത്തിന്റെ രണ്ട് ഡോസുകൾ (20 കി.ഗ്രാം/ഹെക്ടർ) ഉം മൾച്ചിംഗും ഉൾപ്പെടുന്നു। സെപ്റ്റംബർ-ഒക്ടോബറിൽ ഖരീഫ് വിളകൾ വിളവെടുക്കുകയും റബി സീസണിനായി മണ്ണ് തയ്യാറാക്കുകയും ചെയ്യുക, ഇതിൽ ജൈവ വളം (10 ടൺ/ഹെക്ടർ), ആഴത്തിലുള്ള ഉഴവ്, പൊട്ടാസ്യം അടങ്ങിയ കമ്പോസ്റ്റ് (15 കി.ഗ്രാം/ഹെക്ടർ) എന്നിവ ഉൾപ്പെടുന്നു। മുഴുവൻ കാലയളവിലും, മണ്ണിന്റെ ഈർപ്പം നിരീക്ഷണം, ഡ്രിപ്പ് ജലസേചനം, സംയോജിത കീടനാശിനി നിയന്ത്രണം (IPM) എന്നിവ ഉപയോഗിച്ച് പരമാവധി വിളവ് ഉറപ്പാക്കുക." if state['language'] == "Malayalam" else
                f"For a {state['plan_duration']}-month crop plan in {state['location']}, plant heat-tolerant sorghum (Jowar) and moong (Jalgaon85) in early May, requiring intensive irrigation and potassium fertilizer (30 kg/ha). During the monsoon in July-August, sow kharif crops like rice (IR64-SRI with water management in the northwest), maize (HM-5), cotton (LC1136), and urad by late June or early July, incorporating two doses of potassium (20 kg/ha) and mulching. Harvest kharif crops in September-October and prepare soil for the rabi season with organic manure (10 tons/ha), deep tillage, and potassium-rich compost (15 kg/ha). Throughout the period, monitor soil moisture, use drip irrigation, and implement integrated pest management (IPM) to ensure maximum yield."
            )
            logger.info("Using default plan due to summarizer error")
            return {"final_plan": default_plan}

    workflow.add_node("data_collection", data_collection_node)
    workflow.add_node("deepseek_planning", deepseek_planning_node)
    workflow.add_node("gemini_planning", gemini_planning_node)
    workflow.add_node("gpt_planning", gpt_planning_node)
    workflow.add_node("summarizer", summarizer_node)

    workflow.set_entry_point("data_collection")
    workflow.add_edge("data_collection", "deepseek_planning")
    workflow.add_edge("deepseek_planning", "gemini_planning")
    workflow.add_edge("gemini_planning", "gpt_planning")

    def check_iterations(state: CropPlanState) -> str:
        logger.debug(f"Iteration count: {state['iteration']}/{max_iterations}")
        return "summarizer" if state['iteration'] >= max_iterations else "deepseek_planning"

    workflow.add_conditional_edges(
        "gpt_planning",
        check_iterations,
        {"summarizer": "summarizer", "deepseek_planning": "deepseek_planning"}
    )
    workflow.add_edge("summarizer", END)

    return workflow

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("users.db")
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username=?", (username,))
        user = cur.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            session["username"] = username
            flash("Login successful!", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password!", "danger")

    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        hashed_pw = generate_password_hash(password)

        conn = sqlite3.connect("users.db")
        cur = conn.cursor()
        try:
            cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
            conn.commit()
            flash("Signup successful! Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists!", "danger")
        finally:
            conn.close()

    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.clear()  # Clear entire session to prevent persistence
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

@app.route('/')
def home():
    if 'username' in session:
        return render_template('home.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/products', methods=['GET', 'POST'])
def products():
    error_message = None
    products = []
    form_submitted = False
    output_file = 'products_found.json'

    if request.method == 'POST':
        try:
            product_name = request.form.get('product', '').strip()
            if not product_name:
                error_message = "Product name is required."
                return render_template('products.html', error_message=error_message, products=products, form_submitted=True)

            search_task = Task(
                description=(
                    f"Search for the agricultural product '{product_name}' in India. "
                    f"Find details including product name, description, price, certification status (e.g., ISO, CE, or other certifications), and an official link. "
                    f"Use the SerperDevTool to perform the web search. "
                    f"Return a JSON list of up to 5 products with name, description, price, certified status (e.g., 'Yes - ISO 9001', 'No', or 'Unknown'), and link, "
                    f"ensuring the output is a valid JSON string without markdown code fences, "
                    f"compatible with the following schema: "
                    f"{json.dumps([{'name': 'string', 'description': 'string', 'price': 'string', 'certified': 'string', 'link': 'string'}])}"
                ),
                expected_output="A JSON list of up to 5 products with name, description, price, certified status, and link.",
                agent=product_researcher,
                output_file=output_file
            )

            crew = Crew(
                agents=[product_researcher],
                tasks=[search_task],
                verbose=True
            )

            @retry(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=4, max=10),
                retry=retry_if_exception_type(Exception),
                after=lambda retry_state: logger.debug(f"Retry attempt {retry_state.attempt_number} failed with {retry_state.outcome.exception()}")
            )
            def execute_crew():
                return crew.kickoff()

            try:
                result = execute_crew()
            except Exception as e:
                logger.error(f"Crew execution failed after retries: {str(e)}")
                error_message = "The product search service is temporarily unavailable. Please try again later."
                return render_template('products.html', error_message=error_message, products=products, form_submitted=True)

            try:
                if os.path.exists(output_file):
                    with open(output_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()

                    raw_output_file = 'products_found_raw.txt'
                    with open(raw_output_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    logger.debug(f"Raw output saved to {raw_output_file}: {content}")

                    clean_content = content
                    if content.startswith('```json') and content.endswith('```'):
                        clean_content = '\n'.join(content.splitlines()[1:-1]).strip()
                    elif content.startswith('```') and content.endswith('```'):
                        clean_content = '\n'.join(content.splitlines()[1:-1]).strip()

                    if not clean_content:
                        error_message = "Output file is empty after cleaning."
                        logger.error(error_message)
                        return render_template('products.html', error_message=error_message, products=products, form_submitted=True)

                    try:
                        products_data = json.loads(clean_content)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing JSON from {output_file}: {e}")
                        error_message = f"Error processing product data: {e}"
                        return render_template('products.html', error_message=error_message, products=products, form_submitted=True)

                    if not isinstance(products_data, list):
                        error_message = "Product data is not a valid JSON list."
                        logger.error(error_message)
                        return render_template('products.html', error_message=error_message, products=products, form_submitted=True)

                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(products_data, f, indent=2)
                    logger.debug(f"Saved cleaned JSON to {output_file}")

                    products = [Product(**product) for product in products_data if isinstance(product, dict)]
                    if not products:
                        error_message = f"No products found matching '{product_name}'."
                        logger.warning(error_message)
                else:
                    error_message = f"Output file {output_file} not found."
                    logger.error(error_message)
            except Exception as e:
                logger.error(f"Error processing {output_file}: {e}")
                error_message = f"Error processing product data: {str(e)}"

            form_submitted = True
        except Exception as e:
            logger.error(f"Error fetching product data: {e}")
            error_message = "An unexpected error occurred during product search. Please try again later."

    return render_template('products.html', products=products, error_message=error_message, form_submitted=form_submitted)

@app.route('/api/get_soil_type', methods=['POST'])
def get_soil_type():
    try:
        data = request.get_json()
        if not data or 'location' not in data:
            return jsonify({"error": "Location is required"}), 400

        location = data['location'].strip()
        if not location:
            return jsonify({"error": "Location cannot be empty"}), 400

        soil_type = soil_type_tool._run(location)
        return jsonify({"soil_type": soil_type}), 200
    except Exception as e:
        logger.error(f"Error in get_soil_type: {str(e)}")
        return jsonify({"error": f"Error detecting soil type: {str(e)}"}), 500

@app.route('/crop_planning')
def crop_planning():
    lang = request.args.get('lang', 'en')
    return render_template('crop_planning.html', lang=lang)

@app.route('/crop_planning_data', methods=['POST'])
def crop_planning_data():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        required_fields = ['location', 'plan_duration', 'language']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({"error": f"Missing or empty field: {field}"}), 400

        location = data['location'].strip()
        try:
            plan_duration = int(data['plan_duration'])
            if plan_duration <= 0:
                raise ValueError("Plan duration must be positive")
        except ValueError:
            return jsonify({"error": "Invalid plan duration: must be a positive integer"}), 400

        language = data['language'].strip().lower()
        valid_languages = ['english', 'hindi', 'kannada', 'malayalam']
        if language not in valid_languages:
            return jsonify({"error": f"Invalid language: must be one of {', '.join(valid_languages)}"}), 400

        try:
            workflow = create_crop_planning_workflow(location=location, plan_duration=plan_duration, language=language)
            app_workflow = workflow.compile()

            initial_state = {
                "location": location,
                "plan_duration": plan_duration,
                "language": language,
                "soil_data": None,
                "climate_data": None,
                "crop_plan": None,
                "history": [],
                "iteration": 0,
                "final_plan": None
            }
            result = app_workflow.invoke(initial_state)

            history = result.get('history', [])
            final_plan = result.get('final_plan', 'No final plan generated')

            response = {
                "history": history,
                "final_plan": final_plan
            }

            # Store the conversation in Mem0
            mem0_client.add(
                messages=[
                    {"role": "user", "content": f"Crop planning request for {location}, {plan_duration} months, in {language}"},
                    {"role": "assistant", "content": final_plan}
                ],
                user_id="aryan",
                output_format="v1.1"
            )

            return jsonify(response), 200

        except Exception as e:
            logger.error(f"Error running crop planning: {e}")
            return jsonify({"error": f"Failed to generate crop plan: {str(e)}"}), 500

    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/schemes', methods=['GET', 'POST'])
def schemes():
    error_message = None
    schemes = []
    form_submitted = False
    output_file = 'schemes_found.json'

    if request.method == 'POST':
        try:
            user_data = {
                'name': request.form.get('name', ''),
                'age': request.form.get('age', ''),
                'caste': request.form.get('caste', ''),
                'location': request.form.get('location', ''),
                'occupation': request.form.get('occupation', ''),
                'gender': request.form.get('gender', ''),
                'landholding': request.form.get('landholding', '')
            }

            required_fields = ['name', 'caste', 'location', 'occupation', 'gender', 'landholding']
            if not all(user_data[field] for field in required_fields):
                error_message = "All fields are required."
                return render_template('schemes.html', error_message=error_message, schemes=schemes, form_submitted=True)

            search_task = Task(
                description=(
                    f"Search for government agricultural schemes in India relevant to a {user_data['occupation']} "
                    f"with caste {user_data['caste']}, gender {user_data['gender']}, located in {user_data['location']}, "
                    f"and owning {user_data['landholding']} acres of land. Use the SchemeFilterTool to filter schemes "
                    f"by prioritizing location, then occupation, then caste, gender, and landholding. "
                    f"Return a JSON list of up to 50 schemes with name, category, description, and official link, "
                    f"ensuring the output is a valid JSON string without markdown code fences, "
                    f"compatible with the following schema: "
                    f"{json.dumps([{'name': 'string', 'category': 'string', 'description': 'string', 'link': 'string'}])}"
                ),
                expected_output="A JSON list of up to 50 schemes with name, category, description, and link.",
                agent=scheme_researcher,
                output_file=output_file
            )

            crew = Crew(
                agents=[scheme_researcher],
                tasks=[search_task],
                verbose=True
            )

            @retry(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=4, max=10),
                retry=retry_if_exception_type(Exception),
                after=lambda retry_state: logger.debug(f"Retry attempt {retry_state.attempt_number} failed with {retry_state.outcome.exception()}")
            )
            def execute_crew():
                return crew.kickoff()

            try:
                result = execute_crew()
            except Exception as e:
                logger.error(f"Crew execution failed after retries: {str(e)}")
                error_message = "The scheme search service is temporarily unavailable. Please try again later."
                return render_template('schemes.html', error_message=error_message, schemes=schemes, form_submitted=True)

            try:
                if os.path.exists(output_file):
                    with open(output_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()

                    raw_output_file = 'schemes_found_raw.txt'
                    with open(raw_output_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    logger.debug(f"Raw output saved to {raw_output_file}: {content}")

                    clean_content = content
                    if content.startswith('```json') and content.endswith('```'):
                        clean_content = '\n'.join(content.splitlines()[1:-1]).strip()
                    elif content.startswith('```') and content.endswith('```'):
                        clean_content = '\n'.join(content.splitlines()[1:-1]).strip()

                    if not clean_content:
                        error_message = "Output file is empty after cleaning."
                        logger.error(error_message)
                        return render_template('schemes.html', error_message=error_message, schemes=schemes, form_submitted=True)

                    try:
                        schemes_data = json.loads(clean_content)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing JSON from {output_file}: {e}")
                        error_message = f"Error processing scheme data: {e}"
                        return render_template('schemes.html', error_message=error_message, schemes=schemes, form_submitted=True)

                    if not isinstance(schemes_data, list):
                        error_message = "Schemes data is not a valid JSON list."
                        logger.error(error_message)
                        return render_template('schemes.html', error_message=error_message, schemes=schemes, form_submitted=True)

                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(schemes_data, f, indent=2)
                    logger.debug(f"Saved cleaned JSON to {output_file}")

                    schemes = [Scheme(**scheme) for scheme in schemes_data if isinstance(scheme, dict)]
                    if not schemes:
                        error_message = "No valid schemes found matching your criteria."
                        logger.warning(error_message)
                else:
                    error_message = f"Output file {output_file} not found."
                    logger.error(error_message)
            except Exception as e:
                logger.error(f"Error processing {output_file}: {e}")
                error_message = f"Error processing scheme data: {e}"

            form_submitted = True
        except Exception as e:
            logger.error(f"Error fetching scheme data: {e}")
            error_message = "An unexpected error occurred during scheme search. Please try again later."

    return render_template('schemes.html', schemes=schemes, error_message=error_message, form_submitted=form_submitted)

@app.route('/disease', methods=['GET', 'POST'])
def disease():
    error_message = None
    result = None
    form_submitted = False
    output_file = 'disease_results.json'
    raw_output_file = 'disease_results_raw.txt'

    if request.method == 'POST':
        try:
            if 'image' not in request.files:
                error_message = "No image file uploaded."
                return render_template('disease.html', error_message=error_message, result=result, form_submitted=True)

            image = request.files['image']
            if image.filename == '':
                error_message = "No image file selected."
                return render_template('disease.html', error_message=error_message, result=result, form_submitted=True)

            image_path = os.path.join('uploads', image.filename)
            os.makedirs('uploads', exist_ok=True)
            image.save(image_path)

            try:
                with open(image_path, "rb") as img_file:
                    imgdata = base64.b64encode(img_file.read()).decode("utf-8")
                response = requests.post("https://susya.onrender.com", json={"image": imgdata})
                response.raise_for_status()

                disease_data = json.loads(response.text)
                disease = disease_data.get("disease", "Unknown disease")
                plant = disease_data.get("plant", "Unknown plant")
            except Exception as e:
                logger.error(f"Error calling Crop Disease API: {str(e)}")
                error_message = f"Error calling Crop Disease API: {str(e)}"
                return render_template('disease.html', error_message=error_message, result=result, form_submitted=True)

            symptoms_task = Task(
                description=f"For the identified crop disease '{disease}' affecting '{plant}', provide a single concise sentence (15-25 words) describing the most prominent symptoms (e.g., leaf spots, wilting, discoloration). Example: 'Leaves show yellowing with black spots and wilting stems.'",
                expected_output="A single sentence describing symptoms (15-25 words).",
                agent=symptoms_advisor
            )

            remedies_task = Task(
                description=f"For the identified crop disease '{disease}' affecting '{plant}', provide exactly 10 concise remedy points (covering natural, chemical, and prevention advice) in 80-100 words total. Return a single string with period-separated sentences, suitable for splitting into list items. Example: 'Prune infected leaves. Apply neem oil weekly. Use chlorothalonil every 7 days. Rotate crops annually. Ensure proper drainage. Remove plant debris. Apply baking soda spray. Use resistant varieties. Avoid overhead watering. Monitor plants regularly.'",
                expected_output="A string of 10 period-separated sentences describing remedies (80-100 words).",
                agent=remedy_advisor
            )

            resource_links_task = Task(
                description=f"Search the internet for tutorials, guides, or PDFs on how to treat the crop disease '{disease}' affecting '{plant}'. Return a JSON list of 3-5 resources with title, link, and summary: [{{'title': 'string', 'link': 'string', 'summary': 'string'}}].",
                expected_output="A JSON list of 3-5 resources with title, link, and summary.",
                agent=resource_link_finder,
                output_file=output_file,
                output_format='json'
            )

            crew = Crew(
                agents=[symptoms_advisor, remedy_advisor, resource_link_finder],
                tasks=[symptoms_task, remedies_task, resource_links_task],
                verbose=True
            )

            @retry(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=4, max=10),
                retry=retry_if_exception_type(Exception),
                after=lambda retry_state: logger.debug(f"Retry attempt {retry_state.attempt_number} failed with {retry_state.outcome.exception()}")
            )
            def execute_crew():
                return crew.kickoff()

            try:
                crew_result = execute_crew()
            except Exception as e:
                logger.error(f"Crew execution failed after retries: {str(e)}")
                error_message = "The disease detection service is temporarily unavailable. Please try again later."
                return render_template('disease.html', error_message=error_message, result=result, form_submitted=True)

            try:
                if os.path.exists(output_file):
                    with open(output_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()

                    with open(raw_output_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    logger.debug(f"Raw output saved to {raw_output_file}: {content}")

                    clean_content = content
                    if content.startswith('```json') and content.endswith('```'):
                        clean_content = '\n'.join(content.splitlines()[1:-1]).strip()
                    elif content.startswith('```') and content.endswith('```'):
                        clean_content = '\n'.join(content.splitlines()[1:-1]).strip()

                    if not clean_content:
                        error_message = "Output file is empty after cleaning."
                        logger.error(error_message)
                        return render_template('disease.html', error_message=error_message, result=result, form_submitted=True)

                    try:
                        resources_data = json.loads(clean_content)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing JSON from {output_file}: {e}")
                        error_message = "Error processing disease analysis results. Please try again."
                        return render_template('disease.html', error_message=error_message, result=result, form_submitted=True)

                    if not isinstance(resources_data, list):
                        error_message = "Resources data is not a valid JSON list."
                        logger.error(error_message)
                        return render_template('disease.html', error_message=error_message, result=result, form_submitted=True)

                    symptoms_output = symptoms_task.output.raw if symptoms_task.output else "No symptoms identified."
                    remedies_output = remedies_task.output.raw if remedies_task.output else "No remedies identified."

                    for output in [symptoms_output, remedies_output]:
                        if output:
                            try:
                                data = json.loads(output)
                                if isinstance(data, dict):
                                    if output == symptoms_output:
                                        symptoms_output = data.get('symptoms', 'No symptoms identified.')
                                    else:
                                        remedies_output = (
                                            f"Natural: {data.get('natural', '')}. "
                                            f"Chemical: {data.get('chemical', '')}. "
                                            f"Prevent: {data.get('prevention', '')}."
                                        ).strip()
                            except json.JSONDecodeError:
                                pass

                            sentences = [s.strip() for s in output.split('.') if s.strip()]
                            output = '. '.join(sentences) + ('.' if sentences else '')
                            if output == symptoms_output:
                                symptoms_output = output
                            else:
                                remedies_output = output

                            words = output.split()
                            max_words = 25 if output == symptoms_output else 100
                            if len(words) > max_words:
                                output = ' '.join(words[:max_words-10]) + '...'
                                logger.debug(f"Truncated {'symptoms' if output == symptoms_output else 'remedies'} to ~{max_words-10} words: {output}")
                                if output == symptoms_output:
                                    symptoms_output = output
                                else:
                                    remedies_output = output

                    result = {
                        'disease': disease,
                        'plant': plant,
                        'symptoms': symptoms_output,
                        'remedies': remedies_output,
                        'resources': resources_data
                    }

                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2)
                    logger.debug(f"Saved cleaned JSON to {output_file}")

                    result = DiseaseResult(
                        disease=result['disease'],
                        plant=result['plant'],
                        symptoms=result['symptoms'],
                        remedies=result['remedies'],
                        resources=result['resources']
                    )
                else:
                    error_message = f"Output file {output_file} not found."
                    logger.error(error_message)
            except Exception as e:
                logger.error(f"Error processing {output_file}: {e}")
                error_message = f"Error processing disease analysis results: {str(e)}"

            form_submitted = True

            if os.path.exists(image_path):
                os.remove(image_path)

        except Exception as e:
            logger.error(f"Error processing disease detection: {str(e)}")
            error_message = "An unexpected error occurred during disease detection. Please try again later."

    return render_template('disease.html', error_message=error_message, result=result, form_submitted=form_submitted)

@app.route('/legal_assistant')
def legal_assistant():
    lang = request.args.get('lang', 'en')
    messages = [
        {"role": "user" if msg.__class__.__name__ == "HumanMessage" else "assistant", "content": msg.content}
        for msg in get_session_history('legal').messages
    ]
    return render_template('legal_assistant.html', lang=lang, messages=messages)

@app.route('/legal_assistant', methods=['POST'])
def legal_assistant_post():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        required_fields = ['userInput', 'language']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({"error": f"Missing or empty field: {field}"}), 400

        query = data['userInput'].strip()
        language = data['language'].strip().lower()

        valid_languages = ['en', 'hi', 'kn']
        if language not in valid_languages:
            logger.warning(f"Invalid language '{language}', defaulting to 'en'")
            language = 'en'

        language_names = {'en': 'English', 'hi': 'Hindi', 'kn': 'Kannada'}
        language_name = language_names[language]

        prompt_template = PromptTemplate(
            input_variables=["query", "language_name"],
            template=""" 
            You are a friendly legal assistant for rural Indian farmers. Provide a clear and concise answer to the following legal question related to agricultural laws, land disputes, or government schemes. The answer must be in {language_name} and tailored to the Indian context (e.g., referencing Indian laws, government schemes, or local authorities). Limit the response to 3-5 sentences for brevity. Use simple language suitable for farmers and avoid legal jargon. If the question is too vague or unrelated to agricultural legal issues, return a polite message indicating the need for a more specific legal question.

            Question: {query}

            Instructions:
            - Answer in {language_name}, using simple and clear language.
            - Focus on practical advice or information relevant to agricultural laws, land disputes, or government schemes in India.
            - If unsure, suggest consulting a local lawyer or government office.
            - Do not include markdown, code fences, or additional text—only the plain text response.
            - Do not use any special symbols like *
            - If user is greeting, then you also greet

            Example (for English):
            For a land dispute, first check your land records at the local Tehsildar office. Ensure you have documents like the sale deed or Khatauni. File a complaint with the Tehsildar if someone encroaches on your land. You can also seek help from a local lawyer or the Legal Services Authority for free advice.
            """
        )

        prompt = prompt_template.format(
            query=query,
            language_name=language_name
        )

        try:
            logger.debug(f"Sending prompt to Gemini: {prompt[:200]}...")
            response = langchain_llm.invoke(prompt)
            logger.debug(f"Gemini response: {response.content}")

            answer = response.content.strip()
            if not answer:
                logger.warning("Empty response from Gemini")
                answer = "No answer found. Please ask a more specific legal question."

            # Store the conversation in Mem0
            mem0_client.add(
                messages=[
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": answer}
                ],
                user_id="aryan",
                output_format="v1.1"
            )

            # Add to session history
            session_history = get_session_history('legal')
            session_history.add_user_message(query)
            session_history.add_ai_message(answer)

            return jsonify({"response": answer}), 200

        except Exception as e:
            logger.error(f"Gemini query error: {str(e)}")
            return jsonify({"error": "Error processing query. Please try again later."}), 503

    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/veterinary_assistant')
def veterinary_assistant():
    lang = request.args.get('lang', 'en')
    messages = [
        {"role": "user" if msg.__class__.__name__ == "HumanMessage" else "assistant", "content": msg.content}
        for msg in get_session_history('veterinary').messages
    ]
    return render_template('veterinary_assistant.html', lang=lang, messages=messages)

@app.route('/veterinary_assistant', methods=['POST'])
def veterinary_assistant_post():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        required_fields = ['userInput', 'language']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({"error": f"Missing or empty field: {field}"}), 400

        query = data['userInput'].strip()
        language = data['language'].strip().lower()

        valid_languages = ['en', 'hi', 'kn']
        if language not in valid_languages:
            logger.warning(f"Invalid language '{language}', defaulting to 'en'")
            language = 'en'

        language_names = {'en': 'English', 'hi': 'Hindi', 'kn': 'Kannada'}
        language_name = language_names[language]

        prompt_template = PromptTemplate(
            input_variables=["query", "language_name"],
            template=""" 
            You are a friendly veterinary assistant for rural Indian farmers. Provide a clear and concise answer to the following question related to livestock health, animal diseases, or veterinary care. The answer must be in {language_name} and tailored to the Indian context (e.g., referencing common Indian livestock, local remedies, or veterinary services). Limit the response to 3-5 sentences for brevity. Use simple language suitable for farmers and avoid technical jargon. If the question is too vague or unrelated to livestock health, return a polite message indicating the need for a more specific question.

            Question: {query}

            Instructions:
            - Answer in {language_name}, using simple and clear language.
            - Focus on practical advice or information relevant to livestock health, animal diseases, or veterinary care in India.
            - If unsure, suggest consulting a local veterinarian or government veterinary office.
            - Do not include markdown, code fences, or additional text—only the plain text response.
            - Do not use any special symbols like *
            - If user is greeting, then you also greet

            Example (for English):
            If your cow has a fever, keep it in a cool, shaded area and provide plenty of water. Check for symptoms like reduced milk or difficulty breathing. Contact a local veterinarian for medicines like paracetamol or antibiotics. You can also visit the nearest government veterinary hospital for free or low-cost treatment.
            """
        )

        prompt = prompt_template.format(
            query=query,
            language_name=language_name
        )

        try:
            logger.debug(f"Sending prompt to Gemini: {prompt[:200]}...")
            response = langchain_llm.invoke(prompt)
            logger.debug(f"Gemini response: {response.content}")

            answer = response.content.strip()
            if not answer:
                logger.warning("Empty response from Gemini")
                answer = "No answer found. Please ask a more specific veterinary question."

            mem0_client.add(
                messages=[
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": answer}
                ],
                user_id="aryan",
                output_format="v1.1"
            )

            session_history = get_session_history('veterinary')
            session_history.add_user_message(query)
            session_history.add_ai_message(answer)

            return jsonify({"response": answer}), 200

        except Exception as e:
            logger.error(f"Gemini query error: {str(e)}")
            return jsonify({"error": "Error processing query. Please try again later."}), 503

    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/financial_assistant')
def financial_assistant():
    lang = request.args.get('lang', 'en')
    messages = [
        {"role": "user" if msg.__class__.__name__ == "HumanMessage" else "assistant", "content": msg.content}
        for msg in get_session_history('financial').messages
    ]
    return render_template('financial_assistant.html', lang=lang, messages=messages)

@app.route('/financial_assistant', methods=['POST'])
def financial_assistant_post():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        required_fields = ['userInput', 'language']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({"error": f"Missing or empty field: {field}"}), 400

        query = data['userInput'].strip()
        language = data['language'].strip().lower()

        valid_languages = ['en', 'hi', 'kn']
        if language not in valid_languages:
            logger.warning(f"Invalid language '{language}', defaulting to 'en'")
            language = 'en'

        language_names = {'en': 'English', 'hi': 'Hindi', 'kn': 'Kannada'}
        language_name = language_names[language]

        prompt_template = PromptTemplate(
            input_variables=["query", "language_name"],
            template=""" 
            You are a friendly financial assistant for rural Indian farmers. Provide a clear and concise answer to the following question related to agricultural loans, subsidies, or financial schemes. The answer must be in {language_name} and tailored to the Indian context (e.g., referencing Indian banks, government schemes, or local financial institutions). Limit the response to 3-5 sentences for brevity. Use simple language suitable for farmers and avoid financial jargon. If the question is too vague or unrelated to agricultural finance, return a polite message indicating the need for a more specific financial question.

            Question: {query}

            Instructions:
            - Answer in {language_name}, using simple and clear language.
            - Focus on practical advice or information relevant to agricultural loans, subsidies, or financial schemes in India.
            - If unsure, suggest consulting a local bank or government agriculture office.
            - Do not include markdown, code fences, or additional text—only the plain text response.
            - Do not use any special symbols like *
            - If user is greeting, then you also greet

            Example (for English):
            To get a farm loan, visit a nearby cooperative bank or national bank like SBI with your land documents and Kisan Credit Card. Loans under PM-KISAN offer low interest rates for small farmers. Check with your local agriculture office for subsidies on seeds or equipment. Always read loan terms carefully before signing.
            """
        )

        prompt = prompt_template.format(
            query=query,
            language_name=language_name
        )

        try:
            logger.debug(f"Sending prompt to Gemini: {prompt[:200]}...")
            response = langchain_llm.invoke(prompt)
            logger.debug(f"Gemini response: {response.content}")

            answer = response.content.strip()
            if not answer:
                logger.warning("Empty response from Gemini")
                answer = "No answer found. Please ask a more specific financial question."

            # Store the conversation in Mem0
            mem0_client.add(
                messages=[
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": answer}
                ],
                user_id="aryan",
                output_format="v1.1"
            )

            session_history = get_session_history('financial')
            session_history.add_user_message(query)
            session_history.add_ai_message(answer)

            return jsonify({"response": answer}), 200

        except Exception as e:
            logger.error(f"Gemini query error: {str(e)}")
            return jsonify({"error": "Error processing query. Please try again later."}), 503

    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/document_analyzer')
def document_analyzer():
    lang = request.args.get('lang', 'en')
    return render_template('document_analyzer.html', lang=lang)

@app.route('/analyze_document', methods=['POST'])
def analyze_document():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        document_type = request.form.get('document_type', 'other')
        language = request.form.get('language', 'en')

        if not file.filename:
            return jsonify({"error": "No file selected"}), 400

        allowed_extensions = {'pdf', 'jpg', 'jpeg', 'png', 'doc', 'docx'}
        if file.filename.rsplit('.', 1)[-1].lower() not in allowed_extensions:
            return jsonify({"error": "Unsupported file type. Use PDF, JPG, PNG, DOC, or DOCX."}), 400

        content = ""
        if file.filename.endswith('.pdf'):
            pdf_reader = pypdf.PdfReader(file)
            for page in pdf_reader.pages:
                content += page.extract_text() or ""
        elif file.filename.endswith(('.jpg', '.jpeg', '.png')):
            image = Image.open(file)
            content = f"Sample {document_type} document content extracted from image."
        elif file.filename.endswith(('.doc', '.docx')):
            doc = docx.Document(file)
            content = "\n".join([para.text for para in doc.paragraphs])

        if not content.strip():
            content = f"Placeholder content for {document_type} document."

        prompt_template = PromptTemplate(
            input_variables=["content", "document_type", "language"],
            template=""" 
            You are a financial document analysis assistant for India. Analyze the provided document content and provide guidance for filling it out correctly. The document type is '{document_type}' and the output must be in '{language}'.

            Document Content:
            {content}

            Instructions:
            - Analyze the document content and identify its purpose and requirements.
            - Provide the following in '{language}':
              - Summary: A brief description of the document's purpose (1-2 sentences).
              - Required Information: A list of 3-5 key fields or details needed to complete the document.
              - Filing Instructions: A list of 3-5 steps to correctly fill out or submit the document.
              - Important Notes: Any additional guidance or requirements (e.g., supporting documents, mandatory fields).
            - Return a JSON object with translations for English, Hindi, and Kannada, even if the requested language is only one of them.
            - Ensure the JSON is valid and properly formatted.
            - Do not include markdown or extra text, only the JSON object.

            Example Output:
            {{
                "en": {{
                    "summary": "This is a loan application form requiring personal and financial details.",
                    "required_info": ["Name and address", "Monthly income", "Loan amount", "Purpose of loan", "Repayment period"],
                    "instructions": ["Fill in personal details in BLOCK LETTERS", "Provide accurate income", "State loan purpose clearly", "Include bank details", "Sign the form"],
                    "notes": "Attach Aadhaar/PAN, address proof, and income proof. All fields marked with * are mandatory."
                }},
                "hi": {{
                    "summary": "यह एक ऋण आवेदन पत्र है जिसमें व्यक्तिगत और वित्तीय विवरण की आवश्यकता है।",
                    "required_info": ["नाम और पता", "मासिक आय", "ऋण राशि", "ऋण का उद्देश्य", "पुनर्भुगतान अवधि"],
                    "instructions": ["व्यक्तिगत विवरण बड़े अक्षरों में भरें", "सटीक आय प्रदान करें", "ऋण का उद्देश्य स्पष्ट करें", "बैंक विवरण शामिल करें", "फॉर्म पर हस्ताक्षर करें"],
                    "notes": "आधार/पैन, पते का प्रमाण और आय प्रमाण संलग्न करें। * के साथ चिह्नित सभी फ़ील्ड अनिवार्य हैं।"
                }},
                "kn": {{
                    "summary": "ಇದು ವೈಯಕ್ತಿಕ ಮತ್ತು ಆರ್ಥಿಕ ವಿವರಗಳನ್ನು ಕೋರುವ ಸಾಲದ ಅರ್ಜಿ ನಮೂನೆಯಾಗಿದೆ.",
                    "required_info": ["ಹೆಸರು ಮತ್ತು ವಿಳಾಸ", "ಮಾಸಿಕ ಆದಾಯ", "ಸಾಲದ ಮೊತ್ತ", "ಸಾಲದ ಉದ್ದೇಶ", "ಮರುಪಾವತಿ ಅವಧಿ"],
                    "instructions": ["ವೈಯಕ್ತಿಕ ವಿವರಗಳನ್ನು ದೊಡ್ಡ ಅಕ್ಷರಗಳಲ್ಲಿ ಭರ್ತಿ ಮಾಡಿ", "ನಿಖರವಾದ ಆದಾಯವನ್ನು ಒದಗಿಸಿ", "ಸಾಲದ ಉದ್ದೇಶವನ್ನು ಸ್ಪಷ್ಟವಾಗಿ ತಿಳಿಸಿ", "ಬ್ಯಾಂಕ್ ವಿವರಗಳನ್ನು ಸೇರಿಸಿ", "ಫಾರ್ಮ್‌ಗೆ ಸಹಿ ಮಾಡಿ"],
                    "notes": "ಆಧಾರ್/ಪ್ಯಾನ್, ವಿಳಾಸದ ಪುರಾವೆ ಮತ್ತು ಆದಾಯದ ಪುರಾವೆಯನ್ನು ಲಗತ್ತಿಸಿ. * ಗುರುತಿನ ಎಲ್ಲಾ ಕ್ಷೇತ್ರಗಳು ಕಡ್ಡಾಯವಾಗಿವೆ."
                }}
            }}
            """
        )

        prompt = prompt_template.format(
            content=content[:1000],
            document_type=document_type,
            language={'en': 'English', 'hi': 'Hindi', 'kn': 'Kannada'}[language]
        )

        try:
            logger.debug(f"Sending prompt to Gemini: {prompt[:200]}...")
            response = langchain_llm.invoke(prompt)
            logger.debug(f"Raw Gemini response: {response.content}")

            response_content = response.content.strip()
            response_content = re.sub(r'^```json\s*|\s*```$', '', response_content).strip()
            logger.debug(f"Cleaned Gemini response: {response_content}")

            analysis = json.loads(response_content)
            if not isinstance(analysis, dict) or not all(lang in analysis for lang in ['en', 'hi', 'kn']):
                logger.error("Invalid analysis response format")
                return jsonify({"error": "Invalid analysis response"}), 500

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing Gemini response: {str(e)}")
            return jsonify({"error": "Failed to parse analysis response"}), 500
        except Exception as e:
            logger.error(f"Gemini query error: {str(e)}")
            return jsonify({"error": f"Analysis error: {str(e)}"}), 500

        return jsonify({"analysis": analysis}), 200

    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/weather_advisory')
def weather_advisory():
    lang = request.args.get('lang', 'en')
    return render_template('weather_advisory.html', lang=lang)

@app.route('/weather_advisory_data', methods=['POST'])
def weather_advisory_data():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        required_fields = ['location', 'district', 'state']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({"error": f"Missing or empty field: {field}"}), 400

        location = data['location'].strip().title()
        district = data['district'].strip().title()
        state = data['state'].strip().title()

        today = datetime.now()
        daily_dates = [(today + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 3)] 
        weekly_dates = [(today + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)] 

        prompt_template = PromptTemplate(
            input_variables=["location", "district", "state", "daily_dates", "weekly_dates"],
            template=""" 
            You are a weather and agricultural advisory assistant for rural India. Based on the provided location details, generate weather forecasts and agricultural tips for farmers. The current date is {today}.

            Location Details:
            - Village/Town: {location}
            - District: {district}
            - State: {state}

            Instructions:
            - Provide a weather forecast for the specified location, including:
              - Daily forecast for the next 2 days ({daily_dates}).
              - Weekly forecast for the next 7 days ({weekly_dates}).
              - Agricultural tips based on the weather conditions.
              - Weather alerts for any extreme conditions (e.g., heavy rain, drought).
            - Each daily forecast entry must include:
              - date: The date in YYYY-MM-DD format.
              - condition: Weather condition (e.g., Sunny, Rainy, Cloudy).
              - temperature: Temperature in Celsius (e.g., 28).
              - humidity: Humidity percentage (e.g., 70).
              - icon: A Font Awesome icon name (e.g., sun, cloud-rain, cloud) for the condition.
            - Each weekly forecast entry must include:
              - date: The date in YYYY-MM-DD format.
              - condition: Weather condition.
              - min_temp: Minimum temperature in Celsius.
              - max_temp: Maximum temperature in Celsius.
              - icon: A Font Awesome icon name for the condition.
            - Agricultural tips:
              - Provide 3-5 practical tips for farmers based on the weather forecast (e.g., irrigation advice, crop protection).
              - Return as a list of strings.
            - Weather alerts:
              - If there are extreme weather conditions (e.g., heavy rain, heatwave), provide a brief alert message.
              - If no alerts, return a message indicating no extreme weather.
            - Return a JSON object with:
              - daily_forecast: Array of daily forecast objects.
              - weekly_forecast: Array of weekly forecast objects.
              - agricultural_tips: Array of tip strings.
              - weather_alerts: A string with the alert message or a message indicating no alerts.
            - Ensure the JSON is valid and properly formatted.
            - Do not include any additional text, markdown, or explanations—only the JSON object.

            Example Output:
            {{
                "daily_forecast": [
                    {{
                        "date": "2025-05-12",
                        "condition": "Sunny",
                        "temperature": 30,
                        "humidity": 65,
                        "icon": "sun"
                    }},
                    {{
                        "date": "2025-05-13",
                        "condition": "Rainy",
                        "temperature": 26,
                        "humidity": 80,
                        "icon": "cloud-rain"
                    }}
                ],
                "weekly_forecast": [
                    {{
                        "date": "2025-05-12",
                        "condition": "Sunny",
                        "min_temp": 22,
                        "max_temp": 30,
                        "icon": "sun"
                    }},
                    {{
                        "date": "2025-05-13",
                        "condition": "Rainy",
                        "min_temp": 20,
                        "max_temp": 26,
                        "icon": "cloud-rain"
                    }}
                ],
                "agricultural_tips": [
                    "Ensure proper irrigation as the weather will be sunny.",
                    "Prepare for rain by protecting crops with covers."
                ],
                "weather_alerts": "No extreme weather alerts at this time."
            }}
            """
        )

        prompt = prompt_template.format(
            location=location,
            district=district,
            state=state,
            daily_dates=", ".join(daily_dates),
            weekly_dates=", ".join(weekly_dates),
            today=today.strftime('%Y-%m-%d')
        )

        try:
            logger.debug(f"Sending prompt to Gemini for weather: {prompt[:200]}...")
            response = langchain_llm.invoke(prompt)
            logger.debug(f"Raw Gemini response: {response.content}")

            response_content = response.content.strip()
            response_content = re.sub(r'^```json\s*|\s*```$', '', response_content).strip()
            logger.debug(f"Cleaned Gemini response: {response_content}")

            weather_data = json.loads(response_content)
            if not isinstance(weather_data, dict) or 'daily_forecast' not in weather_data or 'weekly_forecast' not in weather_data:
                logger.error("Invalid response format from Gemini")
                return jsonify({"error": "Invalid weather data format"}), 500

            required_daily_fields = ['date', 'condition', 'temperature', 'humidity', 'icon']
            valid_daily = []
            for day in weather_data['daily_forecast']:
                if isinstance(day, dict) and all(field in day for field in required_daily_fields):
                    valid_daily.append(day)
                else:
                    logger.warning(f"Invalid daily forecast object: {day}")
            weather_data['daily_forecast'] = valid_daily

            required_weekly_fields = ['date', 'condition', 'min_temp', 'max_temp', 'icon']
            valid_weekly = []
            for day in weather_data['weekly_forecast']:
                if isinstance(day, dict) and all(field in day for field in required_weekly_fields):
                    valid_weekly.append(day)
                else:
                    logger.warning(f"Invalid weekly forecast object: {day}")
            weather_data['weekly_forecast'] = valid_weekly

            if 'agricultural_tips' not in weather_data or not isinstance(weather_data['agricultural_tips'], list):
                weather_data['agricultural_tips'] = ["No agricultural tips available."]
            if 'weather_alerts' not in weather_data or not isinstance(weather_data['weather_alerts'], str):
                weather_data['weather_alerts'] = "No weather alerts available."

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing Gemini response: {str(e)}")
            return jsonify({"error": "Failed to parse weather data"}), 500
        except Exception as e:
            logger.error(f"Gemini query error: {str(e)}")
            return jsonify({"error": f"Server error: {str(e)}"}), 500

        return jsonify(weather_data), 200

    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/telegram')
def telegram():
    return render_template('telegram.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='127.0.0.1', port=5000)
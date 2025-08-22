from flask import Flask, render_template, request, jsonify
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
# from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


load_dotenv()
app = Flask(__name__)

gemini_api_key = os.getenv('GEMINI_API_KEY')
serper_api_key = os.getenv('SERPER_API_KEY')
mem0_api_key = os.getenv('MEM0_API_KEY')

llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.7,
    api_key=gemini_api_key
)

serper_tool = SerperDevTool(
    api_key=serper_api_key,
    n_results=50
)

class DiseaseResult(BaseModel):
    disease: str
    plant: str
    symptoms: str
    remedies: str
    resources: List[Dict[str, str]]

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
        
crop_disease_tool = CropDiseaseAPI()

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


@app.route('/')
def index():
    return render_template('home.html')



@app.route('/telegram')
def telegram():
    return render_template('telegram.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
import streamlit as st
import os
import glob
import tempfile
import asyncio
# USER_AGENT must be set before any crewai or langchain imports
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"

# Import crewai packages
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from crewai_tools import PDFSearchTool
from crewai_tools import WebsiteSearchTool
from crewai_tools import RagTool
from crewai_tools.tools.rag import RagToolConfig, VectorDbConfig, ProviderSpec
from crewai_tools import FileWriterTool
from crewai_tools import YoutubeVideoSearchTool
from crewai_tools import CodeInterpreterTool
from crewai.tasks.task_output import TaskOutput

# Import langchain packages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
# For RAG pipeline
from langchain_pinecone import PineconeVectorStore, PineconeRerank
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
# For agent tools
from langchain_tavily import TavilySearch, TavilyExtract
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import YouTubeSearchTool
from langchain_community.tools.openai_dalle_image_generation import OpenAIDALLEImageGenerationTool
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.tools import tool

# Import pydantic packages
from pydantic import BaseModel, Field, field_validator, ValidationError
from pathlib import Path
from typing import Type, Optional, Any, Dict, List
import requests

# Import Composio packages
from composio import Composio
from composio_openai_agents import OpenAIAgentsProvider

# Import YOLO model for objection detection
from PIL import Image
# Streamlit has issues with ultralytics YOLO import, so use rfdetr.
#from ultralytics import YOLO
from rfdetr import RFDETRSmall
from rfdetr.util.coco_classes import COCO_CLASSES
# For base64 encoding
import base64 
import io

# Import package for word cloud generation
from wordcloud import WordCloud

# Access secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
tavily_api_key = st.secrets["TAVILY_API_KEY"]
composio_api_key = st.secrets["COMPOSIO_API_KEY"]
composio_user_id = st.secrets["COMPOSIO_USER_ID"]
gemini_api_key = st.secrets["GEMINI_API_KEY"]

# Check for API keys in environment variables
if not openai_api_key or not tavily_api_key or not composio_api_key or not composio_user_id:
    st.warning("Please enter your API keys in the sidebar to proceed.")
    st.stop()

# Set environment variables
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["TAVILY_API_KEY"] = tavily_api_key
os.environ["COMPOSIO_API_KEY"] = composio_api_key
os.environ["COMPOSIO_USER_ID"] = composio_user_id
os.environ['GEMINI_API_KEY'] = gemini_api_key

# --- start of agentic system code ---
# Load LLM
# Using crewai.LLM
llm = LLM(
    model="gpt-4o-mini",
    api_key=openai_api_key,
    temperature=0.2, # lower temp for focused management
    #max_completion_tokens=1000
)

# For Pinecone vector stor RetrievalQA chain, use langchain_openai.ChatOpenAI
qa_llm = ChatOpenAI(
    openai_api_base="https://api.openai.com/v1",
    model_name="gpt-4o-mini",
    api_key=openai_api_key,
    temperature=0.2 # lower temp for focused management
    #max_completion_tokens=1000
)

#vision_llm = LLM(
#    model="gpt-4o",
#    api_key=openai_api_key,
#    temperature=0.2
#)

vision_llm = LLM(
    model="gemini/gemini-2.5-flash-lite",
    api_key=gemini_api_key,
    temperature=0.2
)

# RagTool with ChromaDB as vector store
# default vector db is ChromaDB
vectordb: VectorDbConfig = {
    "provider": "chromadb", # alternative: qdrant
    "config": {
        "collection_name": "bikes_docs",
        "persist_directory": "./my_vector_db"
    }
}

# default embedding model is openai/text-embedding-3-large
embedding_model: ProviderSpec = {
    "provider": "openai",
    "config": {
        "model_name": "text-embedding-3-small"
    }
}

config: RagToolConfig = {
    "vectordb": vectordb,
    "embedding_model": embedding_model,
    "top_k": 4 # default is 4
}

rag_tool = RagTool(
    name="Knowledge Base", # Documentation Tool
    description="""Use this tool to retrieve information from knowledge base about:
    - Folding bikes market outlook 
    - Regulatory requirements by LTA on electric bikes in Singapore
    - Guide on creating a competitive analysis""",
    config=config,
    summarize=True
)

# Add directory of files, use its absolute path
rag_docs_path = os.path.abspath('rag_docs')
rag_tool.add(data_type="directory", path=rag_docs_path)
# Add content from web page
rag_tool.add(data_type="website", url="https://onemotoring.lta.gov.sg/content/onemotoring/home/buying/vehicle-types-and-registrations/PAB.html")
rag_tool.add(data_type="website", url="https://en.wikipedia.org/wiki/Folding_bicycle")

# Tools
class GenerationTool(BaseTool):
    name: str = 'Generation'
    description: str = 'Useful for general queries answered by the LLM.'

    def _run(self, query: str) -> str:
        return llm.invoke(query)

# TavilySearch for general web search
class TavilySearchInput(BaseModel):
    """Input schema for TavilySearchTool"""
    query: str = Field(description="The search query to look up on the internet.")
    search_depth: str = Field(default="basic", description="The depth of the search results to return.")
    include_domains: str = Field(default="", description="A list of domains to include in the search results.")
    include_images: bool = Field(default=False, description="Whether to include images in the search results.")
    include_image_descriptions: bool = Field(default=False, description="Whether to include image descriptions in the search results.")

class TavilySearchTool(BaseTool):
    name: str = "tavily_search"
    description: str = "Searches the internet using Tavily to get real-time information."
    args_schema: Type[BaseModel] = TavilySearchInput
    search: TavilySearch = Field(default_factory=TavilySearch)

    def _run(self, query: str, search_depth: str, include_domains: str, include_images: bool, include_image_descriptions: bool) -> str:
        return self.search.run(query)

class WikipediaTool(BaseTool):
    name: str = "wikipedia"
    description: str = "A tool to search for topics on Wikipedia and return a summary of the article."

    def _run(self, query: str) -> str:
        api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
        return api_wrapper.run(query)

class DallEImageTool(BaseTool):
    name: str = "dalle"
    description: str = "Useful for when you need to generate an image from a text prompt."

    def _run(self, query: str) -> str:
        api_wrapper = DallEAPIWrapper(model="dall-e-3", size="1024x1024")
        return api_wrapper.run(query)

class YouTubeSearchTool(BaseTool):
    name: str = "youtube"
    description: str = "Useful for when you need to search for videos on YouTube."
    search: YouTubeSearchTool = Field(default_factory=YouTubeSearchTool)

    def _run(self, query: str) -> str:
        return self.search.run(query)
    
generation_tool = GenerationTool()
#web_search_tool = FirecrawlSearchTool()
web_search_tool = TavilySearchTool(
    search_depth="basic",
    include_images = True,
    include_image_descriptions = True,
    include_domains = [
        "https://www.brompton.com/stories/design-and-engineering",
        "https://de.dahon.com/pages/technology?srsltid=AfmBOopaKrg-aASd49Nwetbyxas-XzNopsGSVhGln0IIx6IJPi1T39et",
        "https://www.straitstimes.com/paid-press-releases/dahon-v-a-revolutionary-bike-tech-pushing-a-new-frontier-in-green-mobility-20250825",
        "https://www.ternbicycles.com/en/explore/choosing-bike/tern-non-electric-bike-buyer-s-guide",
        "https://www.cyclingnews.com/bikes/reviews/"
    ])

wiki_tool = WikipediaTool() # for quick, general topical overview, as a starting point for research
dalle_tool = DallEImageTool()
youtube_tool = YouTubeSearchTool() # web scraping on YouTube search results page
youtube_rag_tool = YoutubeVideoSearchTool(summarize=True)
file_writer_tool = FileWriterTool(directory='output_files')
code_interpreter = CodeInterpreterTool()

# Composio Reddit
# Initialize Composio toolkits
composio = Composio(provider=OpenAIAgentsProvider(), api_key=composio_api_key)
# Composio Search toolkit, more than one tool
composio_tools = composio.tools.get(user_id=composio_user_id, tools=["reddit"])

class RFDetrInput(BaseModel):
    """Input for RFDetrTool."""
    image_path: str = Field(..., description="URL or local path to the image.")

class RFDetrTool(BaseTool):
    name: str = "RF-DETR Image Analyzer"
    description: str = "Detects objects in images using RF-DETR."
    args_schema: Type[BaseModel] = RFDetrInput

    def _run(self, image_path: str) -> str:
        # Load image and model
        #image = Image.open(requests.get(image_path).content) # URL
        image = Image.open(image_path) # or local path
        model = RFDETRSmall()
        detections = model.predict(image, threshold=0.5)
        # Get the labels by mapping class IDs to class names
        labels = [
            f"{COCO_CLASSES[class_id]}"
            for class_id in detections.class_id
            ]
        # Return label of delected object class
        #return str(results.data)
        return str(labels[0])

#obj_detector_tool = YoloDetectorTool()
obj_detector_tool = RFDetrTool()

# Multimodal agent requires base64 encoding for image data
# As base64 will exceed GPT-4o TPM limit of 30K, lower image resolution before encoding.
class LowResBase64EncodingToolInput(BaseModel):
    image_path: str = Field(..., description="The path to the image file to encode.")
    max_width: int = Field(default=800, description="The maximum width to resize the image to, maintaining aspect ratio.")
    max_height: int = Field(default=800, description="The maximum height to resize the image to, maintaining aspect ratio.")
    quality: int = Field(default=60, description="The quality of the JPEG compression (0-100)."
)

class LowResBase64EncodingTool(BaseTool):
    name: str = 'Image base64 encoding'
    description: str = 'Useful for encoding an image file to a base64 string.'
    args_schema: Type[BaseModel] = LowResBase64EncodingToolInput

    def _run(self, image_path: str, max_width: int = 800, max_height: int = 800, quality: int = 60) -> str:
        # Open and resize image
        img = Image.open(image_path)
        img.thumbnail((max_width, max_height)) # PIL's thumbnail expects a tuple

        # Compress image in memory
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        img_bytes = buffer.getvalue()

        # Base64 Encode
        encoded_string = base64.b64encode(img_bytes).decode('utf-8')
        return encoded_string

encode_image_base64 = LowResBase64EncodingTool()

# Custom tool to generate sentiments word cloud
class WordCloudToolInput(BaseModel):
    text: str = Field(description="The text to generate the word cloud from.")
    colormap: str = Field(description="The color scheme for representing the words.")
    output_image_path: str = Field(description="The path where the word cloud image will be saved.")

class WordCloudGenerationTool(BaseTool):
    name: str = "Word Cloud Generator"
    description: str = "Generates a word cloud image based on input text and saves it to a specified path."
    args_schema: Type[BaseModel] = WordCloudToolInput

    def _run(self, text: str, colormap: str, output_image_path: str) -> str:
        from wordcloud import WordCloud
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=colormap).generate(text)
        wordcloud.to_file(output_image_path)
        return f"Word cloud saved to {output_image_path}"

word_cloud_tool = WordCloudGenerationTool()

# Callback function
def show_word_clouds(output: TaskOutput):
    """Callback function to print task output immediately upon completion."""
    print(f"\n### Sentiment analysis completed ###")
    # Display word clouds
    folderPath = "output_files"
    filesList = glob.glob(folderPath + "/*.png")
    for file in filesList:
        caption = 'Positive feedback word cloud' if 'positive' in file else 'Complaints word cloud'
        st.image(file, caption=caption, width=300)

# Define Pydantic model for structured task output
class MarketAnalysis(BaseModel):
    title: str = Field(description="Title of the market research")
    executive_summary: str = Field(description="Overview of the main points")
    key_trends: List[str] = Field(description="List of identified market trends")
    market_size: str = Field(description="Estimated market size")
    competitors: List[str] = Field(description="Major competitors in the space")
    #competitor_analysis: Optional[Dict[str, Dict[str, str]]] = Field(default_factory=dict, description="Analysis of competitor products")
    competitor_analysis: Dict[str, Any] = Field(default_factory=dict, description="Analysis of competitor products")
    conclusion: str = Field(description="Conclusion or summary of the research")

# Define agents and tasks

# -----------------------------
# Video researcher
# ----------------------------
video_researcher = Agent(
    role="Video Researcher",
    goal="Extract relevant information from YouTube videos",
    backstory='An expert researcher who specializes in analyzing video content.',
    tools=[youtube_rag_tool],
    verbose=True,
)

video_research_task = Task(
    description="""Search for information about the R&D of Brompton's first electric
    folding bike in the YouTube video at {video_url}.""",
    expected_output="""A summary of the R&D strategy underlying Brompton's first 
    electric folding bike mentioned in the video.""",
    #output_file='./output_files/video_transcription.md',
    agent=video_researcher,
)

# -----------------------------
# Reddit researcher
# -----------------------------
reddit_researcher = Agent(
    role="Reddit Search Assistant",
    goal="Help users search Reddit effectively",
    backstory="A helpful assistant with access to Composio Search tools.",
    tools=composio_tools,
    llm=llm,
    verbose=True
)

reddit_search_task = Task(
    description="""Search Reddit forums to get consumer feedback on {product}.""",
    expected_output="Consumer sentiment analysis from Reddit forums",
    #output_file='./output_files/reddit.md',
    agent=reddit_researcher,
)

visualize_sentiments_task = Task(
    description="""Visualize sentiment counts provided by reddit_research task.
    Use the 'WordCloudGenerationTool' to generate word clouds for positive feedback
    and complaints. Focus on the adjectives in the feedback. Specify 'Reds' for 
    the colormap to represent negative feedback. Specify 'PuBuGn' for the colormap 
    to represent positive feedback. The word clouds should be saved in the 
    'output_files' directory.""",
    tools=[word_cloud_tool],
    context=[reddit_search_task],
    expected_output="One word cloud for postive feedback, another word cloud for complaints.",
    agent=reddit_researcher,
    callback=show_word_clouds
)

# -----------------------------
# market researcher
# -----------------------------
analyst = Agent(
    role='Senior Research Analyst',
    goal="""Conduct comprehensive market research about consumer product {product}.""",
    backstory="""An experienced research analyst with expertise in identifying
    market trends and opportunities as well as understanding consumer behavior.
    You always starts with a foundational understanding from Wikipedia before 
    diving deeper. """,
    tools=[rag_tool, wiki_tool, web_search_tool, youtube_tool],
    allow_delegation=True,
    max_iter=10,
    verbose=True,
    llm=llm,
)

market_research_task = Task(
    description="""Research the market for {product}. Include:
    1. Key market trends
    2. Product demand
    3. Market size
    4. Consumer preferences and willingness to pay
    5. Major competitors""",
    expected_output="""Comprehensive, well-structured market research findings that 
    are easy to synthesize into a report""",
    #output_file='output_files/market_research.md',
    agent=analyst
)

# -----------------------------
# writer
# -----------------------------
writer = Agent(
    role='Report Writer',
    goal="""Create comprehensive, well-structured reports combining the provided
    research and news analysis. Do not include any information that is not explicitly
    provided.""",
    backstory="""A professional report writer with experience in business
    intelligence and market analysis. You have an MBA from a top school. You excel
    at synthesizing information into clear and actionable insights.""",
    allow_delegation=True,
    max_iter=10,
    verbose=True,
    llm=llm,
)

writing_task = Task(
    description="""Write a well-researched, market analysis report on consumer
    product {product}.

    Target audience: Product Marketing Manager for {product} company

    Include:
        1. Key market trends
        2. Market size
        3. Any regulatory compliance requirements in Singapore
        4. Major competitors, focusing on the current top 2 competitors
        5. Competitor analysis including SWOT analysis
        6. Comparison table of key features and pricing
        7. Consumer sentiment analysis
        8. Recommendations for {product} company's R&D strategy.

    Collaborate with your teammates to ensure the report is well-researched,
    comprehensive and accurate.

    The report should be approximately 1000-1500 words in length.
    The report content should be reviewed by editor agent.
    """,
    expected_output="""A well-structured and comprehensive report, written in
    Markdown format.""",
    context=[market_research_task],
    #output_file='output_files/draft_report.md',
    agent=writer # writer leads, but can delegate research to researcher
)

# -----------------------------
# editor (for quality assurance)
# -----------------------------
editor = Agent(
    role="Content Editor",
    goal="Ensure content quality and consistency.",
    backstory="""An experienced editor with an eye for detail. You excel at
    critiquing market research and competive analysis reports, ensuring content
    meets high standards for clarity and accuracy.""",
    tools=[rag_tool, wiki_tool, file_writer_tool],
    allow_delegation=True,
    max_iter=10,
    verbose=True,
    llm=llm,
)

editing_task = Task(
    description="""Review and improve the report.

    Target audience: Product Marketing Manager for {product} company

    Your review should:
    1. Check for consistency in tone and style
    2. Improve clarity and readability
    3. Ensure content is comprehensive and accurate
    4. Check that there is a clear takeaway
    5. Check for unintended biases
    6. Check if embedded video links are accessible and not private/deleted

    Use the 'FileWriterTool' to write the final content into a markdown file 
    inside the directory 'output_files'.""",
    context=[writing_task],
    expected_output="A markdown file saved in the 'output_files' directory.",
    output_file='output_files/final_report.md',
    agent=editor
)

# -----------------------------
# Multimodal agent for image analysis and generation
# ----------------------------
# Combine image_analyst and image_artist as one agent
image_analyst = Agent(
    role='Visual Data Specialist',
    goal='Analyze image and provide detailed description or make precise edits.',
    backstory="""An expert in computer vision, capable of interpreting complex
    visual data. You excel at creating descriptive prompts for DALL-E 3.""",
    multimodal=True,
    tools=[encode_image_base64, dalle_tool],
    allow_delegation=False,
    max_iter=10,
    verbose=True,
    llm=vision_llm,
)

# Create a task for both image analysis and generation
generate_image_task = Task(
    description="""Use the 'Base64EncodingTool' to encode the product image at
    {image_url} to a base64 string that you can view. Analyze the image.

    Then use the 'DallEImageTool' to create a photorealistic image
    of a product variant based on the following criteria:

    Only change the color of the product **frame** to {new_color}, maintaining
    all other aspects exactly as they are in the original image.
    """,
    expected_output="""An image URL of a product variant with the frame in {new_color}.""",
    agent=image_analyst,
    result_as_answer=True
)

# -----------------------------
# Input guardrail
# -----------------------------
topic_guard_agent = Agent(
    role='Topic Guardrail Agent',
    goal="""Ensure user questions are related to {topic}.""",
    backstory="""A security expert specialized in ensuring that conversations
    stay on-topic and tasks are within scope. If a question is off-topic, you
    terminate the conversation and stop the crew.""",
    tools=[obj_detector_tool],
    allow_delegation=False,
    verbose=True,
    llm=llm
)

check_topic_task = Task(
    description="""Analyze the user inputs: {product} and {new_color}.
    Determine if {product} is about {topic} AND {new_color} is a valid color.
    Return 'ON_TOPIC' or 'OFF_TOPIC'.""",
    expected_output="A string: 'ON_TOPIC' or 'OFF_TOPIC'",
    agent=topic_guard_agent
)

check_input_image_task = Task(
    description="""Use tool to determine if the image at {image_url} is a bicycle. 
    Return 'BICYCLE' or 'NOT_BICYCLE'.""",
    expected_output="A string: 'BICYCLE' or 'NOT_BICYCLE'",
    agent=topic_guard_agent
)

aggregate_checks_task = Task(
    description="Concatenate the findings from the check_topic and check_input_image tasks.",
    context=[check_topic_task, check_input_image_task],
    expected_output="""A list of two strings: 'ON_TOPIC' or 'OFF_TOPIC', and
    'BICYCLE' or 'NOT_BICYCLE'""",
    agent=topic_guard_agent
)

# Define the crews
guard_crew = Crew(
    agents=[topic_guard_agent],
    tasks=[check_topic_task, check_input_image_task, aggregate_checks_task],
    process=Process.sequential
)

# Marketing crew
crew_1 = Crew(
    agents=[reddit_researcher, analyst, writer, editor],
    tasks=[reddit_search_task, visualize_sentiments_task, market_research_task, writing_task, editing_task],
    process=Process.sequential, # Process.sequential | Process.hierarchical
    #manager_llm=llm, # manager_llm=llm | manager_agent=manager
    planning=True,
    memory=True, # enable memory to keep context
    verbose=False, # True to see collaboration between agents
    output_log_file="output_files/research_crew_log"
)

# A/B testing crew
crew_2 = Crew(
    agents=[image_analyst],
    tasks=[generate_image_task],
    process=Process.sequential,
    verbose=False, # True will output long image base64 encoded string in the log
    output_log_file="output_files/crew_ab_log"
)

# --- end of agentic system code ---

# Validate inputs before passing to crew
class InputValidator(BaseModel):
    product: str
    video_url: str
    image_url: str
    new_color: str
    topic: str

    @field_validator('product', 'video_url', 'image_url', 'new_color', 'topic')
    @classmethod
    def check_not_empty(cls, v) -> str:
        if v is None or len(v.strip()) == 0:
            raise ValueError('Field cannot be null or empty')
        return v
    
    @field_validator('image_url')
    @classmethod
    def check_file_exists(cls, v: str) -> str:
        # Check if the file exists using pathlib
        if not Path(v).exists():
            raise ValueError(f"File not found: {v}")
        return v

# Define function to download image from DALL-E tool
def download_image(image_url, save_path="./generated_image.png"):
    """Downloads an image from a URL and saves it to a file."""
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"Image successfully saved to {save_path}")
        else:
            print(f"Failed to download image. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during download: {e}")

async def run_crew_async(crew: Crew, inputs: dict):
    return crew.kickoff(inputs=inputs)

async def main():
    results = await asyncio.gather(
        crew_1.kickoff_async(inputs=validated_data.dict()),
        crew_2.kickoff_async(inputs=validated_data.dict())
    )

    st.divider()
    st.success("✅ Task Completed!")
    st.markdown("### ✨ Results:")

    for i, result in enumerate(results, 1):
        if i==1:
            st.write(f"Crew {i} Result:", result.raw)
            # Display sentiment word clouds
            folderPath = "output_files"
            filesList = glob.glob(folderPath + "/*.png")
            for file in filesList:
                caption = 'Positive feedback word cloud' if 'positive' in file else 'Complaints word cloud'
                st.image(file, caption=caption, width=300)

        if i == 2:
            st.divider()
            download_image(result, save_path=f"output_files/generated_variant_{new_color}.jpg")
            if os.path.exists(f"output_files/generated_variant_{new_color}.jpg"):
                st.write(f"Crew {i} Result:")
                st.image(f"output_files/generated_variant_{new_color}.jpg", caption='Product variant image', width=200)
            else:
                st.warning("Image download failed.")


# --- PAGE CONFIG ---
#st.set_page_config(page_title="AI-powered Market Research Assistant", page_icon=":bike:", layout="wide")
#st.title(":bike: AI-powered Market Research Assistant")
icon_img = Image.open("bicycle_icon.png")
st.set_page_config(layout="wide")
col1, col2 = st.columns([1, 10])
with col1:
    st.image(icon_img, width=50)
with col2:
    #st.title("AI-powered Market Research Assistant")
    st.markdown("<h2 style='margin-top: 0px;'>AI-powered Market Research Assistant</h2>", unsafe_allow_html=True)
st.markdown("Hi, enter a folding bike product/brand, and let me help you with the market research.")

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ User inputs")
    product = st.text_input("Product:", placeholder="e.g., Tern folding bike")
    #video_url = st.text_input("Video link for analysis:", value="https://www.youtube.com/watch?v=lhDoB9rGbGQ")
    video_url = "https://www.youtube.com/watch?v=lhDoB9rGbGQ"
    folderPath = os.path.abspath('input_files')
    filesList = glob.glob(folderPath + "/*")
    basenames = [os.path.basename(f) for f in filesList]
    selected_image = st.selectbox("Select product image for color variation:", 
                             options=basenames, index=0)
    image_url = os.path.join(folderPath, selected_image)
    if image_url:
        st.sidebar.image(image_url, caption='Product original image', width=200)
    new_color = st.text_input("New color for product variant:", placeholder="e.g., white, blue, gold, red, green")
    
    st.divider()
    st.info("Version v0.2.0")

if st.button("Run Task"):
    if not product:
        st.error("Please enter a product name.")
    elif not new_color:
        st.error("Please enter a new color.")
    else:
        # Remove all existing files in output_files folder
        folderPath = "output_files"
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        else:
            # Get list of all the files in the folder
            filesList = glob.glob(folderPath + "/*")
            for file in filesList:
                os.remove(file)

        # Validate inputs before passing to crew
        raw_data = {"product": product,
            "video_url": video_url,
            "image_url": image_url,
            "new_color": new_color,
            "topic": "Folding bicycles" # folding bikes, electric bikes
        }

        try:
            validated_data = InputValidator(**raw_data)
            # Proceed with crew execution
            with st.spinner("Analyzing and executing task..."):
                # Run guardrail
                 #result = guard_crew.kickoff(inputs=validated_data.dict())
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(run_crew_async(guard_crew, inputs=validated_data.dict()))

                # Check for termination condition
                if "OFF_TOPIC" in result.raw:
                    st.warning(f"Session terminated: please check your inputs.")
                else:
                    # Proceed with main agents
                    if "NOT_BICYCLE" in result.raw:
                        st.warning(f"Please select a bicycle image. Skipping variant generation for now.")
                        # Run crew_1 only
                        #result = crew_1.kickoff(inputs=validated_data.dict())
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        result = loop.run_until_complete(run_crew_async(crew_1, inputs=validated_data.dict()))
                    else:
                        # Run both crews asynchronously
                        #await main() # async cannot be outside async function
                        asyncio.run(main())

        except ValidationError as e:
            st.warning(f"Validation Error: {e}")
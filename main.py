import streamlit as st
import os
import glob
import tempfile
import asyncio
# USER_AGENT must be set before any crewai or langchain imports
#os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"

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
from typing import Type, Optional
import requests

# Import Composio packages
from composio import Composio
from composio_openai_agents import OpenAIAgentsProvider

# --- PAGE CONFIG ---

st.set_page_config(page_title="AI-powered Market Research Assistant", page_icon="🦋", layout="wide")
st.title("🦋 AI-powered Market Research Assistant")
st.markdown("Hi, enter a folding bike product name, and let me help you with the market research.")

# Access secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
tavily_api_key = st.secrets["TAVILY_API_KEY"]
composio_api_key = st.secrets["COMPOSIO_API_KEY"]
composio_user_id = st.secrets["COMPOSIO_USER_ID"]

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ User inputs")
    product = st.text_input("Product name:", placeholder="e.g., Tern folding bike")
    youtube_video_url = st.text_input("Video link for analysis:", value="https://www.youtube.com/watch?v=lhDoB9rGbGQ")
 
    image_url = st.text_input("Product image for color variation:", value="input_files/Tern-Verge-D9-black.jpg")
    #uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    #image_url = uploaded_image.name if uploaded_image else ""
    '''
    # As uploaded file is temporarily held in server RAM, we need to save it to a temp file for use with functions that require a file path
    if uploaded_image:
        # Create a temporary directory and file
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, uploaded_image.name)
            with open(path, "wb") as f:
                f.write(uploaded_image.getvalue())
            image_url = path
    
            if image_url:
                st.sidebar.image(image_url, caption='Product original image', width=200)
            new_color = st.text_input("New color for product variant:", placeholder="e.g., white, blue, gold, red, green")
    '''
    st.divider()
    st.info("Version v0.2.0")

# Check for API keys in environment variables
# Check for keys
if not openai_api_key or not tavily_api_key or not composio_api_key or not composio_user_id:
    st.warning("Please enter your API keys in the sidebar to proceed.")
    st.stop()

# Set environment variables
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["TAVILY_API_KEY"] = tavily_api_key
os.environ["COMPOSIO_API_KEY"] = composio_api_key
os.environ["COMPOSIO_USER_ID"] = composio_user_id

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

vision_llm = LLM(
    model="gpt-4o-mini",
    api_key=openai_api_key,
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
rag_tool.add(data_type="website", url="https://en.wikipedia.org/wiki/List_of_bicycle_parts")

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
        "https://www.cyclingnews.com/reviews/"
    ])

wiki_tool = WikipediaTool() # for quick, general topical overview, as a starting point for research
dalle_tool = DallEImageTool()
youtube_tool = YouTubeSearchTool() # web scraping on YouTube search results page
#youtube_rag_tool = YoutubeVideoSearchTool(youtube_video_url='https://www.youtube.com/watch?v=lhDoB9rGbGQ', summarize=True)
youtube_rag_tool = YoutubeVideoSearchTool(summarize=True)
file_writer_tool = FileWriterTool(directory='output_files')
code_interpreter = CodeInterpreterTool()

# Composio Reddit
# Initialize Composio toolkits
composio = Composio(provider=OpenAIAgentsProvider(), api_key=composio_api_key)
# Composio Search toolkit, more than one tool
composio_tools = composio.tools.get(user_id=composio_user_id, tools=["reddit"])

# Define agents and tasks
video_researcher = Agent(
    role="Video Researcher",
    goal="Extract relevant information from YouTube videos",
    backstory='You have a strong background in analyzing video content.',
    tools=[youtube_rag_tool],
    verbose=True,
)

reddit_researcher = Agent(
    role="Reddit Search Assistant",
    goal="Help users search Reddit effectively",
    backstory="You are a helpful assistant with access to Composio Search tools.",
    tools=composio_tools,
    llm=llm,
    verbose=True,
    max_iter=5,
)

market_researcher = Agent(
    role='Market Researcher',
    goal="Conduct comprehensive market research about the assigned product.",
    backstory="""You are an experienced market analyst with expertise in identifying 
    market trends and opportunities as well as understanding consumer behavior.""",
    tools=[rag_tool, web_search_tool, wiki_tool, youtube_tool],
    allow_delegation=True,
    max_iter=5,
    llm=llm,
    verbose=True
)

writer = Agent(
    role='Writer',
    goal="""Create a clear competitive analysis report for the assigned product that
    is actionable and provides valuable insights to the business owner.""",
    backstory="""You are a seasoned product marketing manager who understands the
    intersection of product strategy, customer insights and go-to-market execution.
    You have a gift for combining market research and competitive analysis to find
    a competitive advantage for consumer product companies. You are able to explain
    complex concepts in accessible language.""",
    tools=[rag_tool, file_writer_tool],
    allow_delegation=True,
    max_iter=5,
    llm=llm,
    verbose=True
)

content_reviewer = Agent(
    role="Content Reviewer and Editor",
    goal="Ensure content is accurate, well-structured, and with clear takeaways.",
    backstory="""You are a meticulous editor with an MBA and years of experience
    reviewing market reports by consultants. You have an eye for clarity and coherence.
    You excel at improving content, while maintaining the original author's voice
    and ensuring consistent quality across multiple sections of the report.""",
    tools=[youtube_rag_tool, file_writer_tool],
    allow_delegation=True,
    max_iter=5,
    llm=llm,
    verbose=True
)

topic_guard_agent = Agent(
    role='Topic Guardrail Agent',
    goal='Ensure all user questions are strictly related to {specific_topic}.',
    backstory="""You are a security expert specialized in ensuring that conversations
    stay on-topic and tasks are within scope. If a question is off-topic, you
    terminate the conversation and stop the crew.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# -----------------------------
# Tasks
# -----------------------------
video_research_task = Task(
    description="""Search for information about the R&D of Brompton's first electric
    folding bike in the YouTube video at {youtube_video_url}, and provide a
    comprehensive summary of the main points.""",
    expected_output="""A detailed summary of the R&D strategy behind Brompton's
    first electric folding bike from the video.""",
    #output_file='./output_files/youtube_video.md',
    agent=video_researcher,
)

reddit_search_task = Task(
    description='Search Reddit forums to get consumer feedback on a product.',
    expected_output="A helpful response addressing the user's request",
    #output_file='./output_files/reddit.md',
    agent=reddit_researcher,
)

market_research_task = Task(
    description="""Research the market for {product}. Include:
    1. Key market trends
    2. Product demand
    3. Market size
    4. Consumer preferences and willingness to pay
    5. Major competitors""",
    expected_output="A well-structured, comprehensive report in Markdown format.",
    context=[video_research_task, reddit_search_task],
    #output_file='output_files/market_research.md',
    agent=market_researcher
)

writing_task = Task(
    description="""Write a competitive analysis report on consumer product {product}.

    Target audience: Product Marketing Manager for {product} company

    Your content should:
    - Be based on the current top 2 competitors.
    - Include SWOT analysis.
    - Highlight any regulatory compliance requirements in Singapore.
    - Outline the common consumer complaints about {product}.
    - Provide concrete recommendations for {product} company's R&D strategy.
    - Be approximately 800-1000 words in length.
    - Include relevant image and video links.

    """,
    expected_output="A well-structured, comprehensive report in Markdown format.",
    context=[video_research_task, market_research_task],
    #output_file='./output_files/draft_report.md',
    agent=writer
)

review_task = Task(
    description="""Review and improve the competitive analysis report.

    Target audience: Product Marketing Manager for {product} company

    Your review should:
    1. Check for consistency in tone and style
    2. Improve clarity and readability
    3. Ensure content is comprehensive and accurate
    4. Use the 'YoutubeVideoSearchTool' to verify the validity of any image or YouTube video links
    5. Check for bias and suggest improvements

    """,
    expected_output="""'An improved, polished version of the report that
    maintains the original structure but enhances clarity, accuracy and consistency.""",
    context=[writing_task],
    agent=content_reviewer
)

check_topic_task = Task(
    description="""Analyze the user input: {product}. Determine if it is about {specific_topic}.
    Return 'ON_TOPIC' or 'OFF_TOPIC'.""",
    expected_output='A string: "ON_TOPIC" or "OFF_TOPIC"',
    agent=topic_guard_agent
)

# Create a multimodal agent which comes pre-configured with AddImageTool to process images
image_analyst = Agent(
    role="Visual Quality Inspector",
    goal="Analyze product images and provide accurate and detailed descriptions",
    backstory="Senior quality control expert with expertise in visual inspection of consumer products.",
    multimodal=True,
    llm=vision_llm,
    verbose=True
)

image_artist = Agent(
    role='Senior Industrial Designer',
    goal='Create accurate, detailed visual representations of a specific topic',
    backstory="""You are a veteran product industrial designer with a knack for
    creating innovative design solutions that balance visuals, user-interaction,
    performance, manufacturing, and cost. You specialize in creating detailed
    prompts for DALL-E 3.""",
    tools=[dalle_tool],
    allow_delegation=False,
    max_iter=10,
    llm=vision_llm,
    verbose=True
)

# Create a task for image analysis
describe_image_task = Task(
    description="""Analyze the product image at {image_url} and provide a detailed description""",
    expected_output="An accurate and detailed description of the product image",
    #output_file='./output_files/image_description.md',
    agent=image_analyst
)

# Create a task for image generation
generate_image_task = Task(
    description="""Create a photorealistic image of a variant of the product image
    at {image_url}. Change the color of the product **frame** to {new_color},
    while maintaining the other attributes of the product. You must not modify
    any other part of the image.""",
    expected_output="""An image URL of the product variant with the **frame** in
    {new_color}.""",
    context=[describe_image_task],
    agent=image_artist,
    result_as_answer=True
)

# Define the crews
guard_crew = Crew(
    agents=[topic_guard_agent],
    tasks=[check_topic_task],
    process=Process.sequential
)

# Marketing crew
crew_1 = Crew(
    agents=[video_researcher, reddit_researcher, market_researcher, content_reviewer],
    tasks=[video_research_task, reddit_search_task, market_research_task, writing_task, review_task],
    process=Process.hierarchical, # Process.sequential | Process.hierarchical
    manager_llm=llm, # manager_llm=llm | manager_agent=manager
    planning=True,
    memory=True, # enable memory to keep context
    verbose=False,
    #output_log_file="output_files/crew_mkt_log"
)

# A/B testing crew
crew_2 = Crew(
    agents=[image_analyst, image_artist],
    tasks=[describe_image_task, generate_image_task],
    process=Process.sequential,
    planning=False,
    verbose=False,
    #output_log_file="output_files/crew_ab_log"
)

# --- end of agentic system code ---

# Validate inputs before passing to crew
class InputValidator(BaseModel):
    product: str
    youtube_video_url: str
    image_url: str
    new_color: str
    specific_topic: str

    @field_validator('product', 'youtube_video_url', 'image_url', 'new_color', 'specific_topic')
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

async def run_crew_async(crew: Crew, inputs: dict):
    return crew.kickoff(inputs=inputs)

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

async def main():
    results = await asyncio.gather(
        crew_1.kickoff_async(inputs=validated_data.dict()),
        crew_2.kickoff_async(inputs=validated_data.dict())
    )

    st.divider()
    st.success("✅ Task Completed!")
    st.markdown("### ✨ Results:")

    for i, result in enumerate(results, 1):
        st.write(f"Crew {i} Result:", result.raw)
        if i == 2:
            #st.write(f"Generated Image URL: {result}")
            download_image(result, save_path=f"output_files/generated_variant_{new_color}.jpg")
            # Display generated image from URL
            st.image(f"output_files/generated_variant_{new_color}.jpg", caption='Product variant image', width=200)


# --- Start of run code ---

# Remove all existing files in output_files folder
folderPath = "output_files"
if not os.path.exists(folderPath):
    os.makedirs(folderPath)
else:
    # Get list of all the files in the folder
    filesList = glob.glob(folderPath + "/*")
    for file in filesList:
        #print("Removing file {}".format(file))
        os.remove(file)

if st.button("Run Task"):
    # Validate inputs before passing to crew
    raw_data = {"product": product,
                "youtube_video_url": youtube_video_url,
                "image_url": image_url,
                "new_color": new_color,
                "specific_topic": "Bicycles" # folding bikes, electric bikes
    }
    try:
        validated_data = InputValidator(**raw_data)
        # Proceed with crew execution
        with st.spinner("Analyzing and executing task..."):
            # Run guardrail
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(run_crew_async(guard_crew, inputs=validated_data.dict()))

            # Check for termination condition
            if "OFF_TOPIC" in result.raw:
                st.warning(f"Session terminated: {product} is off-topic and not allowed.")
            else:
                # Proceed with main agents
                #await main() # async cannot be outside async function
                asyncio.run(main())

    except ValidationError as e:
        st.warning(f"Validation Error: {e}")

'''
Market research assistant, with RAG knowledge base pre-loaded with folding/e-bikes, and Streamlit frontend
Version v0
'''
# Install dependencies
# !pip install crewai crewai_tools langchain_community langchain_openai langchain_pinecone langchain-tavily composio composio-openai-agents python-dotenv gdown requests
#!pip install wikipedia youtube_search

# Load environment variables
import warnings
warnings.filterwarnings("ignore")
import os
from dotenv import load_dotenv
load_dotenv()

# Import crewai packages
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from crewai_tools import PDFSearchTool
from crewai_tools import WebsiteSearchTool
#from crewai_tools import FirecrawlSearchTool
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

# Load the LLM
# Using crewai.LLM
llm = LLM(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.2, # lower temp for focused management
    #max_completion_tokens=1000
)

# For Pinecone vector stor RetrievalQA chain, use langchain_openai.ChatOpenAI
qa_llm = ChatOpenAI(
    openai_api_base="https://api.openai.com/v1",
    model_name="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.2 # lower temp for focused management
    #max_completion_tokens=1000
)

vision_llm = LLM(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
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
rag_docs_path = os.path.abspath('./rag_docs')
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
#web_search_tool = TavilySearchTool()
#web_search_tool = FirecrawlSearchTool()
# "https://dahon.com/technology" (has pop-up)
web_search_tool = TavilySearchTool(
    search_depth="basic",
    include_images = True,
    include_image_descriptions = True,
    include_domains = [
        "https://www.brompton.com/stories/design-and-engineering",
        "https://de.dahon.com/pages/technology?srsltid=AfmBOopaKrg-aASd49Nwetbyxas-XzNopsGSVhGln0IIx6IJPi1T39et",
        "https://www.straitstimes.com/paid-press-releases/dahon-v-a-revolutionary-bike-tech-pushing-a-new-frontier-in-green-mobility-20250825",
        "https://www.ternbicycles.com/en/explore/choosing-bike/tern-non-electric-bike-buyer-s-guide"
    ])

wiki_tool = WikipediaTool() # for quick, general topical overview, as a starting point for research
dalle_tool = DallEImageTool()
youtube_tool = YouTubeSearchTool() # web scraping on YouTube search results page
#youtube_rag_tool = YoutubeVideoSearchTool(youtube_video_url='https://www.youtube.com/watch?v=lhDoB9rGbGQ', summarize=True)
youtube_rag_tool = YoutubeVideoSearchTool(summarize=True) # semantic search within content of Youtube video
#pdf_search_tool = PDFSearchTool(pdf='https://onemotoring.lta.gov.sg/content/dam/onemotoring/Buying/PDF/PAB/List_of_Approved_PAB_Models.pdf')
#lta_website_tool = WebsiteSearchTool(website='https://onemotoring.lta.gov.sg/content/onemotoring/home/buying/vehicle-types-and-registrations/PAB.html')
#dir_search_tool = DirectorySearchTool(directory='./ebike_docs')
file_writer_tool = FileWriterTool(directory='./output_files')
code_interpreter = CodeInterpreterTool()

# Composio Reddit
# Initialize Composio toolkits
composio = Composio(provider=OpenAIAgentsProvider(), api_key=os.getenv('COMPOSIO_API_KEY'))
# Composio Search toolkit, more than one tool
composio_tools = composio.tools.get(user_id=os.getenv('COMPOSIO_USER_ID'), tools=["reddit"])

# Define agents and tasks
video_researcher = Agent(
    role="Video Researcher",
    goal="Extract relevant information from YouTube videos",
    backstory='You are an expert researcher who specializes in analyzing video content.',
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
    goal="""Conduct comprehensive market research and in-depth competitive analysis
    about the assigned product.""",
    backstory="""You have years of experience conducting market research and helping
    consumer goods companies understand product demand and improve marketing efforts.
    You excel at analyzing consumer preferences, behaviors, and competitor strategies.
    """,
    tools=[rag_tool, web_search_tool, wiki_tool, youtube_tool],
    allow_delegation=False,
    max_iter=5,
    llm=llm,
    verbose=True
)

# product marketing strategy / competitive analysis report
writer = Agent(
    role='Writer',
    goal="""Create a clear competitive analysis report for the assigned product that
    is actionable and provides valuable insights to the business owner.""",
    backstory="""You are a seasoned product marketing manager who understands the
    intersection of product strategy, customer insights and go-to-market execution.
    You have a gift for combining market research and competitive analysis to find
    a competitive advantage for consumer product companies. You are able to explain
    complex concepts in accessible language.""",
    tools=[file_writer_tool],
    allow_delegation=False,
    max_iter=5,
    llm=llm,
    verbose=True
)

content_reviewer = Agent(
    role="Content Reviewer and Editor",
    goal="""Ensure content is accurate, comprehensive, well-structured, and insightful
    with clear takeaways.""",
    backstory="""You are a meticulous editor with an MBA and years of experience
    reviewing market reports by consultants. You have an eye for clarity and coherence.
    You excel at improving content, while maintaining the original author's voice
    and ensuring consistent quality across multiple sections in the report.""",
    tools=[youtube_tool, file_writer_tool],
    allow_delegation=False,
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
    human_input=False,
    output_file='./output_files/youtube_video.md',
    agent=video_researcher,
)

reddit_search_task = Task(
    description='Search Reddit forums to get consumer feedback on a product.',
    expected_output="A helpful response addressing the user's request",
    output_file='./output_files/reddit.md',
    agent=reddit_researcher,
)

market_research_task = Task(
    description="""Conduct market research and product competitive analysis for
    the consumer product {product}.""",
    expected_output="""Comprehensive, accurate and unbiased competitive analysis
    that is useful for formulating a product strategy""",
    context=[reddit_search_task],
    human_input=False,
    output_file='./output_files/market_research.md',
    agent=market_researcher
)

# product strategy / competitive analysis report
writing_task = Task(
    description="""Write a competitive analysis report for consumer product {product}.

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
    expected_output="""A full-fledged report with appropriate headings, lists and 
    emphasis. Formatted as markdown.""",
    context=[video_research_task, market_research_task],
    human_input=False,
    output_file='./output_files/draft_report.md',
    agent=writer
)

review_task = Task(
    description="""Review and improve the competitive analysis report.

    Target audience: Product Marketing Manager for {product} company

    Your review should:
    - Fix any grammatical or spelling errors
    - Fix any broken image or video links
    - Improve clarity and readability
    - Ensure content is comprehensive and accurate
    - Enhance the structure and flow
    - Add any missing information
    """,
    expected_output="""'An improved, polished version of the Markdown report that
    maintains the original structure but enhances clarity, accuracy and consistency.""",
    context=[writing_task],
    human_input=False,
    output_file='./output_files/final_report.md',
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
    role="Image Analyst",
    goal="Analyze product images and provide accurate and detailed descriptions",
    backstory="Expert in visual product analysis with deep knowledge of design and features",
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
    output_file='./output_files/image_description.md',
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
    output_log_file="./output_files/crew_mkt_log"
)

# A/B testing crew
crew_2 = Crew(
    agents=[image_analyst, image_artist],
    tasks=[describe_image_task, generate_image_task],
    process=Process.sequential,
    planning=False,
    verbose=False,
    output_log_file="./output_files/crew_ab_log"
)
    

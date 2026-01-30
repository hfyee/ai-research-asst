import os
import glob
import streamlit as st
# USER_AGENT must be set before any crewai or langchain imports
#os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"
from crewai import Crew, Process
import asyncio
# Import pydantic packages
from pydantic import BaseModel, Field, field_validator, ValidationError
from pathlib import Path
from typing import Type, Optional
import requests
import re

import crew_logic
from crew_logic import guard_crew, crew_1, crew_2

st.set_page_config(page_title="AI-powered Market Research Assistant", page_icon="🦋", layout="wide")
st.title("🦋 AI-powered Market Research Assistant")
st.markdown("Hi, enter a folding bike product name, and let me help you with the market research.")

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ User inputs")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    tavily_api_key = st.text_input("Tavily API Key", type="password")
    composio_api_key = st.text_input("Composio API Key", type="password")
    composio_user_id = st.text_input("Composio User ID", type="password")
    st.divider()
    product = st.text_input("Product name:", placeholder="e.g., Tern folding bike")
    youtube_video_url = st.text_input("Video link for analysis:", value="https://www.youtube.com/watch?v=lhDoB9rGbGQ")
    #uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    image_url = st.text_input("Product image for color variation:", value="input_files/Tern-Verge-D9-black.jpg")
    if image_url:
        st.sidebar.image(image_url, caption='Product original image', width=200)
    new_color = st.text_input("New color for product variant:", placeholder="e.g., white, blue, gold, red, green")
    st.divider()
    st.info("Version v0.2.0")

# Check for API keys in environment variables
# Check for keys
if not openai_api_key or not tavily_api_key or not composio_api_key or not composio_user_id:
    st.warning("Please enter your API keys in the sidebar to proceed.")
    st.stop()

# Set Environment Variables for LangChain
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["TAVILY_API_KEY"] = tavily_api_key
os.environ["COMPOSIO_API_KEY"] = composio_api_key
os.environ["COMPOSIO_USER_ID"] = composio_user_id

# Custom classes & functions
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
            st.image(f"output_files/generated_variant_{new_color}.jpg", caption='Product variant imageL', width=200)


# Start of run code

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

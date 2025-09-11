import os
import io
import base64
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from PIL import Image
import requests
import re

# Initialize FastAPI app
app = FastAPI(title="Math Bot API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBb-eK-BUDNxV0WyrTHdRw6QkJJMgUAOYw")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize the model (using gemini-pro for text, gemini-pro-vision for images)
text_model = genai.GenerativeModel('gemini-2.5-pro')
vision_model = genai.GenerativeModel('gemini-2.5-pro')

# Initialize a separate classification model with temperature for deterministic results
classification_generation_config = genai.types.GenerationConfig(
    temperature=0.1,  # Low temperature for deterministic outputs
    max_output_tokens=1,  # Just one token to classify the message
)

classification_model = genai.GenerativeModel(
    'gemini-2.5-pro',
    generation_config=classification_generation_config
)

def process_math_problem(prompt: str, image_data=None):
    """
    Process a math problem using Gemini API
    """
    try:
        if image_data:
            # Process image with Gemini Vision
            response = vision_model.generate_content([prompt, image_data])
        else:
            # Process text with Gemini Pro
            response = text_model.generate_content(prompt)
        
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing with Gemini: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Math Bot API is running", "version": "1.0.0"}

@app.post("/solve/text")
async def solve_text_problem(problem: str = Form(...)):
    """
    Solve a math problem from text input
    """
    prompt = f"""
    Solve the following math problem. Provide step-by-step reasoning and box the final answer.
    Use LaTeX for mathematical expressions where appropriate, enclosing them in $.
    
    Problem: {problem}
    
    Important: Always format your response with clear steps and box the final answer.
    """
    
    try:
        solution = process_math_problem(prompt)
        return JSONResponse(content={"problem": problem, "solution": solution})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/solve/image")
async def solve_image_problem(file: UploadFile = File(...)):
    """
    Solve a math problem from an image
    """
    # Check if the file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image file
        image_data = await file.read()
        
        # Create prompt for the vision model
        prompt = """
        Extract and solve the math problem from this image. Provide step-by-step reasoning and box the final answer.
        Use LaTeX for mathematical expressions where appropriate, enclosing them in $.
        
        Important: Always format your response with clear steps and box the final answer.
        """
        
        # Process the image
        solution = process_math_problem(prompt, Image.open(io.BytesIO(image_data)))
        
        return JSONResponse(content={"solution": solution})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/solve/url")
async def solve_image_url(url: str = Form(...)):
    """
    Solve a math problem from an image URL
    """
    try:
        # Download image from URL
        response = requests.get(url)
        response.raise_for_status()
        
        # Open image
        image_data = Image.open(io.BytesIO(response.content))
        
        # Create prompt for the vision model
        prompt = """
        Extract and solve the math problem from this image. Provide step-by-step reasoning and box the final answer.
        Use LaTeX for mathematical expressions where appropriate, enclosing them in $.
        
        Important: Always format your response with clear steps and box the final answer.
        """
        
        # Process the image
        solution = process_math_problem(prompt, image_data)
        
        return JSONResponse(content={"solution": solution})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image URL: {str(e)}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
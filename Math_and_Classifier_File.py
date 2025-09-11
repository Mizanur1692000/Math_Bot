import os
import io
import base64
import sys
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import google.generativeai as genai
from PIL import Image
import requests
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Math Bot and Message Classification API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY environment variable is not set.")
    print("Please create a .env file with your API key:")
    print("GEMINI_API_KEY=your_api_key_here")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)

# Initialize models for math solving
text_model = genai.GenerativeModel('gemini-2.5-flash')

# Initialize the classification model with deterministic settings
classification_generation_config = genai.types.GenerationConfig(
    temperature=0.1,  # Low temperature for consistent outputs
    max_output_tokens=500,  # Adjust based on expected response length
)

classification_model = genai.GenerativeModel(
    'gemini-2.5-flash',
    generation_config=classification_generation_config
)

def process_math_problem(prompt: str, image_data=None):
    """
    Process a math problem using Gemini API with natural output formatting
    """
    try:
        if image_data:
            # For vision tasks, we need to use the correct model
            vision_model = genai.GenerativeModel('gemini-2.5-flash')
            response = vision_model.generate_content([prompt, image_data])
        else:
            # Process text with Gemini Pro
            response = text_model.generate_content(prompt)
        
        # Return the raw response without excessive formatting
        return response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing with Gemini: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def root():
    # Create static directory if it doesn't exist
    os.makedirs("static", exist_ok=True)
    
    # Serve the index.html file
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        # Fallback HTML if file doesn't exist
        return """
        <html>
            <head>
                <title>Math Bot & Message Classifier</title>
                <meta http-equiv="refresh" content="0; URL='/static/index.html'" />
            </head>
            <body>
                <p>Redirecting to <a href="/static/index.html">the application</a>...</p>
            </body>
        </html>
        """

@app.post("/solve/text")
async def solve_text_problem(problem: str = Form(...)):
    """
    Solve a math problem from text input
    """
    prompt = f"""
    Solve the following math problem. Provide a clear, step-by-step solution in natural language.
    Use plain text formatting without markdown or excessive symbols.
    
    Problem: {problem}
    
    Format your response with clear steps and a final answer.
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
        Extract and solve the math problem from this image. Provide a clear, step-by-step solution in natural language.
        Use plain text formatting without markdown or excessive symbols.
        
        Format your response with clear steps and a final answer.
        """
        
        # Process the image
        img = Image.open(io.BytesIO(image_data))
        solution = process_math_problem(prompt, img)
        
        return JSONResponse(content={"solution": solution})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/solve/image-with-prompt")
async def solve_image_with_prompt(
    file: UploadFile = File(...),
    prompt: str = Form(...)
):
    """
    Solve a math problem from an image with a custom user prompt
    """
    # Check if the file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image file
        image_data = await file.read()
        
        # Process the image
        img = Image.open(io.BytesIO(image_data))
        solution = process_math_problem(prompt, img)
        
        return JSONResponse(content={"solution": solution, "user_prompt": prompt})
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
        Extract and solve the math problem from this image. Provide a clear, step-by-step solution in natural language.
        Use plain text formatting without markdown or excessive symbols.
        
        Format your response with clear steps and a final answer.
        """
        
        # Process the image
        solution = process_math_problem(prompt, image_data)
        
        return JSONResponse(content={"solution": solution})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image URL: {str(e)}")

@app.post("/classify")
async def classify_message(message: str = Form(...)):
    """
    Classify a message to detect inappropriate content
    Returns 1 if content is problematic, 0 if normal
    """
    classification_prompt = f"""
    Analyze the following message and classify it. Return only a single digit (0 or 1) with no additional text.
    
    Return 1 if the message contains any of the following:
    - Bullying or harassment
    - Slang or inappropriate language
    - Dangerous links (malware, viruses, etc.)
    - Phishing attempts
    - Any other harmful content
    
    Return 0 if the message is normal, safe conversation.
    
    Message: {message}
    
    Classification:
    """
    
    try:
        response = classification_model.generate_content(classification_prompt)
        
        # Extract just the classification digit
        classification = re.search(r'\d', response.text)
        
        if classification:
            result = int(classification.group(0))
            return JSONResponse(content={"message": message, "classification": result})
        else:
            # If no digit found, default to safe (0)
            return JSONResponse(content={"message": message, "classification": 0})
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error classifying message: {str(e)}")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
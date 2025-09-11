import os
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from google.generativeai import types
import re

# Initialize FastAPI app
app = FastAPI(title="Message Classification API", version="1.0.0")

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

# Initialize the classification model with deterministic settings
generation_config = types.GenerationConfig(
    temperature=0.1,  # Low temperature for consistent outputs
    max_output_tokens=500,  # Adjust based on expected response length
)

classification_model = genai.GenerativeModel(
    'gemini-2.5-pro',
    generation_config=generation_config
)

@app.get("/")
async def root():
    return {"message": "Message Classification API is running", "version": "1.0.0"}

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

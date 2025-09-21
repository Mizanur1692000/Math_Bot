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
import json
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
text_model = genai.GenerativeModel('gemini-2.5-pro')

# Initialize the classification model with deterministic settings
classification_generation_config = genai.types.GenerationConfig(
    temperature=0.1,  # Low temperature for consistent outputs
    max_output_tokens=500,  # Adjust based on expected response length
)

classification_model = genai.GenerativeModel(
    'gemini-2.5-pro',
    generation_config=classification_generation_config
)

def process_math_problem(prompt: str, image_data=None):
    """
    Process a math problem using Gemini API.
    Accepts:
      - image_data: either a PIL.Image.Image OR raw bytes (will be converted to PIL.Image)
      - or None for text-only prompts
    Returns the model's text response (stripped).
    """
    try:
        if image_data:
            # If bytes were passed, convert to PIL Image
            if isinstance(image_data, (bytes, bytearray)):
                img = Image.open(io.BytesIO(image_data))
            elif isinstance(image_data, Image.Image):
                img = image_data
            else:
                # If it's some other wrapper from SDK, try to use it directly
                img = image_data

            # Use the vision-capable model and pass the PIL Image object
            vision_model = genai.GenerativeModel('gemini-2.5-pro')
            response = vision_model.generate_content([prompt, img])
        else:
            # Text-only
            response = text_model.generate_content(prompt)

        return response.text.strip()
    except Exception as e:
        # Keep the HTTPException so FastAPI returns 500 with the error
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
    Use LaTeX formatting for mathematical expressions (enclose in $ for inline and $$ for display equations).
    Avoid markdown formatting except for LaTeX.
    
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
        Use LaTeX formatting for mathematical expressions (enclose in $ for inline and $$ for display equations).
        Avoid markdown formatting except for LaTeX.
        
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
    file: UploadFile = File(None),
    prompt: str = Form(None)
):
    """
    Solve a math problem from an image with an optional custom user prompt.
    Accepts:
      - image only (use default prompt to extract & solve)
      - prompt only (solve from text prompt)
      - both image and prompt (use prompt + image)
    At least one of `file` or `prompt` must be provided.
    """
    # Require at least one input
    if file is None and (prompt is None or prompt.strip() == ""):
        raise HTTPException(status_code=400, detail="Provide an image file, a text prompt, or both.")

    try:
        img = None

        # If a file is provided, validate and open it
        if file is not None:
            if not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="File must be an image")
            image_bytes = await file.read()
            img = Image.open(io.BytesIO(image_bytes))

        # If no prompt provided but image is present, use a sensible default prompt
        if not prompt or prompt.strip() == "":
            prompt = """
            Extract and solve the math problem from this image. Provide a clear, step-by-step solution in natural language.
            Use LaTeX formatting for mathematical expressions (enclose in $ for inline and $$ for display equations).
            Avoid markdown formatting except for LaTeX.

            Format your response with clear steps and a final answer.
            """

        # Call the shared processing function:
        # - if img is None, process_math_problem(prompt) -> text-only
        # - if img present, process_math_problem(prompt, img) -> vision + prompt
        solution = process_math_problem(prompt, img) if img is not None else process_math_problem(prompt)

        # Build response
        response_content = {
            "solution": solution,
            "input": {
                "provided_image": bool(img),
                "provided_prompt": bool(prompt and prompt.strip() != "")
            }
        }
        if file is not None:
            response_content["input"]["image_filename"] = file.filename
        if prompt is not None:
            response_content["user_prompt"] = prompt

        return JSONResponse(content=response_content)

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
        Use LaTeX formatting for mathematical expressions (enclose in $ for inline and $$ for display equations).
        Avoid markdown formatting except for LaTeX.
        
        Format your response with clear steps and a final answer.
        """
        
        # Process the image
        solution = process_math_problem(prompt, image_data)
        
        return JSONResponse(content={"solution": solution})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image URL: {str(e)}")
@app.post("/check-solution")
async def check_solution(
    problem_text: str = Form(None),
    problem_file: UploadFile = File(None),
    solution_text: str = Form(None),
    solution_file: UploadFile = File(None)
):
    """
    Check if the uploaded solution matches the correct solution for a given problem.
    - problem_text / problem_file: one or both may be provided (at least one required).
    - solution_text / solution_file: one or both may be provided (at least one required).
    Returns JSON including:
      - comparison: 0 if correct, 1 if incorrect (or unclear)
      - correct_solution: canonical final answer produced from the problem
      - extracted_solution: final answer extracted from user's solution input
      - raw_result: the raw text returned by the model when comparing
    """
    # Validate inputs: at least one problem input and one solution input
    if (not problem_text or not problem_text.strip()) and problem_file is None:
        raise HTTPException(status_code=400, detail="Provide the problem as text, an image, or both.")
    if (not solution_text or not solution_text.strip()) and solution_file is None:
        raise HTTPException(status_code=400, detail="Provide the solution as text, an image, or both.")

    # Validate image content types (if provided)
    if problem_file is not None and not problem_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Problem file must be an image.")
    if solution_file is not None and not solution_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Solution file must be an image.")

    try:
        # --- Read and cache file bytes once (if provided), convert to PIL.Image ---
        problem_bytes = None
        problem_img = None
        if problem_file is not None:
            problem_bytes = await problem_file.read()
            try:
                # convert to RGB to avoid mode-related issues and validate
                problem_img = Image.open(io.BytesIO(problem_bytes)).convert("RGB")
                problem_img.load()
            except Exception:
                raise HTTPException(status_code=400, detail="Could not open problem image. Ensure it's a valid image file.")

        solution_bytes = None
        solution_img = None
        if solution_file is not None:
            solution_bytes = await solution_file.read()
            try:
                solution_img = Image.open(io.BytesIO(solution_bytes)).convert("RGB")
                solution_img.load()
            except Exception:
                raise HTTPException(status_code=400, detail="Could not open solution image. Ensure it's a valid image file.")

        # --- 1) Produce the canonical correct solution (final answer only) ---
        problem_prompt = """
Solve the following math problem. Provide only the final answer in its simplest form.
Use LaTeX formatting if appropriate. Do not include any explanations or steps.

"""
        if problem_text and problem_text.strip():
            problem_prompt += f"Problem (text): {problem_text.strip()}\n\n"
        problem_prompt += "Final answer only:"

        # Pass the PIL.Image object (if present) — do NOT pass raw bytes to the SDK
        if problem_img is not None:
            correct_solution = process_math_problem(problem_prompt, problem_img)
        else:
            correct_solution = process_math_problem(problem_prompt)

        correct_solution = correct_solution.strip()

        # --- 2) Compare user's provided solution with canonical correct_solution ---
        check_prompt_base = f"""
Extract the final answer from the provided solution (either text or image). Then compare it with the correct answer: {correct_solution}

Return ONLY a single word: CORRECT (if the answers match, considering equivalent formats like fractions vs decimals) or INCORRECT (if they don't match).
If you cannot determine, return INCORRECT.
Do not include any explanations.
"""
        # If user provided solution text, include it
        if solution_text and solution_text.strip():
            check_prompt_base = f"Solution (text): {solution_text.strip()}\n\n" + check_prompt_base

        # Run the comparison: pass PIL.Image if available
        if solution_img is not None:
            raw_result = process_math_problem(check_prompt_base, solution_img)
        else:
            raw_result = process_math_problem(check_prompt_base)

        raw_result = raw_result.strip()

        # Determine verdict
        m = re.search(r'\b(CORRECT|INCORRECT)\b', raw_result, re.IGNORECASE)
        if m:
            verdict = m.group(1).upper()
            comparison = 0 if verdict == "CORRECT" else 1
        else:
            comparison = 1  # treat unclear as incorrect per your requirement

        # --- 3) Extract final answer from user's solution for transparency (reuse PIL.Image) ---
        extract_prompt = """
Extract the final answer from the provided solution. Return only the answer (use LaTeX if appropriate).
If you cannot determine the final answer, return "UNCLEAR".
"""
        if solution_text and solution_text.strip():
            extract_prompt = f"Solution (text): {solution_text.strip()}\n\n" + extract_prompt

        if solution_img is not None:
            extracted_raw = process_math_problem(extract_prompt, solution_img)
        else:
            extracted_raw = process_math_problem(extract_prompt)

        extracted_solution = extracted_raw.strip() if extracted_raw else ""

        return JSONResponse(content={
            "comparison": comparison,          # 0 == correct, 1 == incorrect / unclear
            "correct_solution": correct_solution,
            "extracted_solution": extracted_solution,
            "raw_result": raw_result,
            "inputs": {
                "problem_text_provided": bool(problem_text and problem_text.strip()),
                "problem_image_provided": bool(problem_file),
                "solution_text_provided": bool(solution_text and solution_text.strip()),
                "solution_image_provided": bool(solution_file)
            }
        })

    except HTTPException:
        # re-raise FastAPI HTTPExceptions
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


import logging
logging.basicConfig(level=logging.INFO)

def _extract_text_from_genai_response(res) -> str:
    """
    Robust extraction for google.generativeai responses.
    Tries .text, result.parts, candidates[].content.parts, top-level parts, then falls back to str(res).
    """
    # If it's already a string
    if isinstance(res, str):
        return res

    # 1) safe .text (catch if accessor raises)
    try:
        txt = getattr(res, "text", None)
        if isinstance(txt, str) and txt.strip():
            return txt
    except Exception:
        # .text accessor can raise for complex responses; ignore and continue
        pass

    # 2) result.parts -> join
    result = getattr(res, "result", None)
    if result is not None:
        parts = getattr(result, "parts", None)
        if parts:
            out = []
            for p in parts:
                t = getattr(p, "text", None)
                if t:
                    out.append(t)
            if out:
                return "".join(out)

        # result.candidates[].content.parts
        candidates = getattr(result, "candidates", None)
        if candidates:
            for cand in candidates:
                content = getattr(cand, "content", None)
                if content:
                    parts = getattr(content, "parts", None)
                    if parts:
                        out = [getattr(p, "text", "") for p in parts if getattr(p, "text", None)]
                        if any(out):
                            return "".join(out)
                if getattr(cand, "text", None):
                    return cand.text

    # 3) top-level candidates / outputs / choices
    candidates = getattr(res, "candidates", None) or getattr(res, "outputs", None) or getattr(res, "choices", None)
    if candidates:
        for cand in candidates:
            content = getattr(cand, "content", None)
            if content:
                parts = getattr(content, "parts", None)
                if parts:
                    out = [getattr(p, "text", "") for p in parts if getattr(p, "text", None)]
                    if any(out):
                        return "".join(out)
            if getattr(cand, "text", None):
                return cand.text
            if getattr(cand, "output_text", None):
                return cand.output_text

    # 4) top-level parts
    parts = getattr(res, "parts", None)
    if parts:
        out = [getattr(p, "text", "") for p in parts if getattr(p, "text", None)]
        if any(out):
            return "".join(out)

    # 5) fallback: log repr and return str(res)
    try:
        logging.info("Unrecognized GenAI response shape; repr(response)=%s", repr(res))
    except Exception:
        pass
    return str(res)


@app.post("/classify")
async def classify_message(message: str = Form(...)):
    """
    Classify a message: return 1 if harmful, 0 otherwise.
    Robust to Gemini SDK response shapes.
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
        # call the classification model (you created earlier with generation_config)
        response = classification_model.generate_content(classification_prompt)

        # robustly extract text from response
        raw = _extract_text_from_genai_response(response).strip()
        logging.info("Classify raw text: %s", raw)

        # 1) prefer a standalone digit 0 or 1 (not part of other digits)
        m = re.search(r'(?<!\d)([01])(?!\d)', raw)
        if m:
            classification = int(m.group(1))
            return JSONResponse(content={"message": message, "classification": classification})

        # 2) accept words "zero" / "one"
        raw_lower = raw.lower()
        if "one" in raw_lower and "zero" not in raw_lower:
            return JSONResponse(content={"message": message, "classification": 1})
        if "zero" in raw_lower and "one" not in raw_lower:
            return JSONResponse(content={"message": message, "classification": 0})

        # 3) fallback: default to 0 (safe) per your requirement
        return JSONResponse(content={"message": message, "classification": 0})

    except Exception as e:
        # log the exception for debugging and return 500
        logging.exception("Error in classify endpoint")
        raise HTTPException(status_code=500, detail=f"Error classifying message: {str(e)}")


MAX_QUESTIONS = 20  # safety limit

@app.post("/generate-question")
async def generate_math_question(
    grade: str = Form(...),
    subject: str = Form(...),
    count: int = Form(1)  # default is 1 if not provided
):
    """
    Generate one or more math questions based on grade level and subject/topic.
    Uses multiple strategies to reliably return `count` question/answer pairs.
    """
    try:
        # sanitize count
        try:
            count = int(count)
        except Exception:
            count = 1
        if count < 1:
            count = 1
        if count > MAX_QUESTIONS:
            count = MAX_QUESTIONS

        # Primary prompt: ask for strict JSON only
        json_prompt = f"""
You are a math teacher. Generate {count} unique math questions for a student in grade {grade}
on the topic of {subject}. Each question should be age-appropriate, clear, and solvable.

Return **only** valid JSON. The JSON must be an array of objects with exactly these keys:
[
  {{
    "question": "question text here",
    "answer": "answer text here"
  }},
  ...
]

Do NOT include any additional text outside the JSON array. Make sure there are exactly {count} objects.
"""
        response = text_model.generate_content(json_prompt)
        text = response.text.strip()

        questions = []

        # 1) Try to extract JSON array from response
        json_match = re.search(r'(\[.*\])', text, re.DOTALL)
        if json_match:
            try:
                arr = json.loads(json_match.group(1))
                # normalize and add
                for item in arr:
                    q = item.get("question") if isinstance(item, dict) else None
                    a = item.get("answer") if isinstance(item, dict) else None
                    if q and a:
                        questions.append({"question": q.strip(), "answer": a.strip()})
            except Exception:
                # JSON parse failed, fall through to regex parsing
                pass

        # 2) If JSON didn't yield enough, try to parse using Q/A regex
        if len(questions) < count:
            # Find all Question ... Answer ... pairs using robust regex
            qa_pairs = re.findall(
                r"(?:Question\s*\d*[:：]\s*)(.*?)(?:\r?\n\s*Answer\s*\d*[:：]\s*)(.*?)(?=(?:\r?\n\s*Question\s*\d*[:：])|$)",
                text,
                re.DOTALL | re.IGNORECASE
            )
            for q, a in qa_pairs:
                if len(questions) >= count:
                    break
                questions.append({"question": q.strip(), "answer": a.strip()})

        # 3) If still short, request additional individual questions until we reach `count`
        attempt = 0
        while len(questions) < count and attempt < (count * 2):
            attempt += 1
            remaining = count - len(questions)
            single_prompt = f"""
Generate 1 unique math question for grade {grade} on the topic {subject}.
Return as:
Question: ...
Answer: ...
Do not repeat previous questions.
"""
            resp = text_model.generate_content(single_prompt)
            text_single = resp.text.strip()

            # try to parse single pair
            m = re.search(r"Question\s*\d*[:：]\s*(.*?)(?:\r?\n\s*Answer\s*\d*[:：]\s*(.*))?$",
                          text_single, re.DOTALL | re.IGNORECASE)
            if m:
                q = m.group(1).strip()
                a = m.group(2).strip() if m.group(2) else ""
                if q and a:
                    # Avoid duplicates (basic check)
                    if not any(q == existing["question"] for existing in questions):
                        questions.append({"question": q, "answer": a})
                        continue

            # fallback: try to split by lines if model returned short text
            lines = [ln.strip() for ln in text_single.splitlines() if ln.strip()]
            if len(lines) >= 2:
                q = lines[0]
                a = lines[1]
                if not any(q == e["question"] for e in questions):
                    questions.append({"question": q, "answer": a})

            # if nothing added in this loop, continue and retry (up to attempt limit)

        # final safety: if still short, pad with placeholders
        while len(questions) < count:
            questions.append({
                "question": "Unable to generate question — please retry.",
                "answer": ""
            })

        # normalize numbering and return exactly `count` items
        result = []
        for i in range(count):
            qitem = questions[i]
            result.append({
                "number": i + 1,
                "question": qitem["question"],
                "answer": qitem["answer"]
            })

        return JSONResponse(content={
            "grade": grade,
            "subject": subject,
            "count": count,
            "questions": result
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating questions: {str(e)}")


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
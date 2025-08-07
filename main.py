# main.py - Minimal version for debugging Railway deployment
import os
from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="HackRx Test API",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"status": "online", "message": "Basic deployment successful"}

@app.get("/api/v1/health")
async def health():
    return {
        "status": "healthy",
        "environment_vars": {
            "gemini_key_present": bool(os.getenv("GEMINI_API_KEY")),
            "hackrx_token_present": bool(os.getenv("HACKRX_TOKEN"))
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

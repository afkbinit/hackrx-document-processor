# main.py - Minimal test version
from fastapi import FastAPI
import uvicorn
import os

app = FastAPI(title="Railway Test")

@app.get("/")
async def root():
    return {"status": "Railway deployment successful", "message": "Basic FastAPI working"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

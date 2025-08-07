from fastapi import FastAPI
import uvicorn
import os

app = FastAPI(title="HackRx API Test")

@app.get("/")
async def root():
    return {"status": "online", "message": "Railway deployment successful"}

@app.get("/api/v1/health") 
async def health():
    return {"status": "healthy"}

@app.get("/api/v1/hackrx/run", methods=["POST"])
async def hackrx_webhook():
    return {"message": "Webhook endpoint working"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

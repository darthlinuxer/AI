from fastapi import FastAPI
import logging
import uvicorn
from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from otel_config import setup_tracing
from rag_routes import router as rag_routes

app = FastAPI()
setup_tracing()

app.include_router(rag_routes)

FastAPIInstrumentor.instrument_app(app)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.get("/")
async def index() -> str:
    message = "Server is online!"
    logger.info(message)
    return message

def run_server():
    try:
        uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
    except Exception as e:
        print(f"Error starting server: {e}")
    finally:
        print("Server stopped")


if __name__ == "__main__":
    run_server()
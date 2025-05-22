from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.api import router

app = FastAPI()

# allow all domains cross-origin (recommended for development environment, specify domains for production environment)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # allow all frontend domains
#     allow_credentials=True,
#     allow_methods=["*"],  # allow all methods
#     allow_headers=["*"],  # allow all request headers
# )

app.include_router(router)

# uvicorn main:app --reload --host=0.0.0.0 --port=112233
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8211, workers=3)

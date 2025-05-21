from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.api import router

app = FastAPI()
#app.include_router(router)

# 允许所有域名跨域（开发环境推荐，生产环境建议指定域名）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有前端域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有请求头
)

app.include_router(router)

# uvicorn main:app --reload --host=0.0.0.0 --port=112233
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8211, workers=3)

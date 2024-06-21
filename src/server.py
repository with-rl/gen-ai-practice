import sys
import os
import pathlib
import argparse
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langserve import add_routes

sys.path.append(pathlib.Path(__file__).resolve().parent)
from chain import FAISSChain


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


def get_args():
    p = argparse.ArgumentParser()

    p.add_argument("--hf_token", type=str, required=True)
    p.add_argument("--vdb_index", type=str, default="data/faiss_kowiki_vdb")
    p.add_argument(
        "--question_embedding_id",
        type=str,
        default="snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    )
    p.add_argument(
        "--llm_model_id",
        type=str,
        default="google/gemma-1.1-7b-it",
    )

    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--port", type=int, default=18080)

    args = p.parse_args()
    return args


def main(args):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = args.hf_token
    add_routes(
        app,
        FAISSChain(args).chain,
        path="/question",
        enable_feedback_endpoint=True,
        enable_public_trace_link_endpoint=True,
        playground_type="default",
    )


if __name__ == "__main__":
    import uvicorn

    args = get_args()
    main(args)

    uvicorn.run(app, host="0.0.0.0", port=args.port)


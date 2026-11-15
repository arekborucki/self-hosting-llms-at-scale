"""Chapter 8, listing 8.1: a round-robin router for two vLLM servers.

Start the servers on ports 8001 and 8002, then:  python3 rr_proxy.py
"""
import itertools

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from starlette.background import BackgroundTask

BACKENDS = itertools.cycle(["http://localhost:8001",
                            "http://localhost:8002"])
app = FastAPI()
client = httpx.AsyncClient(timeout=None)


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/v1/models")
async def models():
    r = await client.get("http://localhost:8001/v1/models")
    return Response(r.content, media_type="application/json")


@app.post("/v1/{path:path}")
async def forward(path: str, request: Request):
    url = f"{next(BACKENDS)}/v1/{path}"
    req = client.build_request("POST", url, content=await request.body(),
                               headers={"content-type": "application/json"})
    r = await client.send(req, stream=True)
    return StreamingResponse(r.aiter_raw(), status_code=r.status_code,
                             media_type=r.headers.get("content-type"),
                             background=BackgroundTask(r.aclose))


uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

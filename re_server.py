import math
import time
import logging
import threading
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import cv2
from PIL import Image, ImageDraw, ImageFont

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
    "https://florr.io"  # Ensure this origin is included
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


class Item(BaseModel):
    textList: List[dict]


@app.post("/textlist")
async def receive_text_list(content: Item):
    global data
    for item in content.textList:
        if item["size"] == 12:
            data = item
            canvas_mid = (data["canvasWidth"] // 2, data["canvasHeight"] // 2)
            petal_point = (data["x"], data["y"])
            degree = math.atan2(
                petal_point[1] - canvas_mid[1], petal_point[0] - canvas_mid[0]) / math.pi
            print(f"Degree: {degree}")
    return {"status": "success", "received": data}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000,
                log_level="critical", access_log=False)

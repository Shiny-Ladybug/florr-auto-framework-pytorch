import json
import math
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import sys
import os
import pynput
import jsonlines
import threading
from ultralytics import YOLO
import numpy as np
import pyautogui
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from rich.console import Console
from datetime import datetime

console = Console()

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
MAX_MOBS_NUM = 10

model = YOLO("./models/stf_det.pt")


def one_hot_encode_mob(mob_name: str):
    MOB_TYPE_ONE_HOTS = ["Starfish", "Jellyfish", "Bubble"]
    index_ = {mob: idx for idx, mob in enumerate(MOB_TYPE_ONE_HOTS)}
    index = index_[mob_name]
    return [1 if i == index else 0 for i in range(len(MOB_TYPE_ONE_HOTS))]


def check_health():
    def get_color(img, x, y):
        return img.getpixel((x, y))

    def is_empty(rgb: tuple):
        if rgb[0] <= 50:
            if rgb[1] <= 50:
                if rgb[2] <= 50:
                    return True
        return False

    img = pyautogui.screenshot(region=[0, 0, 1920, 1080])
    health_num = 4
    left_4 = 116
    right_96 = 278
    last = 278
    color = get_color(img=img, x=288, y=108)
    if not is_empty(color):
        return {"percent": 100, "florr": 32041}
    for i in range(right_96, left_4, -1):
        color = get_color(img=img, x=i, y=97)
        if not is_empty(color):
            last = i
            break
    death = get_color(img, 105, 110)
    if is_empty(death):
        return {"percent": 0, "florr": 0}
    delta = last-left_4
    k = (96-4)/(right_96-left_4)
    health_num = delta*k+4
    health_num = round(health_num, 2)
    health_florr = round(health_num*34954*0.01, 2)
    return {"percent": health_num, "florr": health_florr}


def check_health():
    def get_color(img, x, y):
        return img.getpixel((x, y))

    def is_empty(rgb: tuple):
        if rgb[0] <= 50:
            if rgb[1] <= 50:
                if rgb[2] <= 50:
                    return True
        return False

    img = pyautogui.screenshot(region=[0, 0, 1920, 1080])
    health_num = 4
    left_4 = 116
    right_96 = 278
    last = 278
    color = get_color(img=img, x=288, y=108)
    if not is_empty(color):
        return {"percent": 100, "florr": 32041}
    for i in range(right_96, left_4, -1):
        color = get_color(img=img, x=i, y=97)
        if not is_empty(color):
            last = i
            break
    death = get_color(img, 105, 110)
    if is_empty(death):
        return {"percent": 0, "florr": 0}
    delta = last-left_4
    k = (96-4)/(right_96-left_4)
    health_num = delta*k+4
    health_num = round(health_num, 2)
    health_florr = round(health_num*34954*0.01, 2)
    return {"percent": health_num, "florr": health_florr}


def state2dataset(state: dict):
    health_norm = state["health"]/100
    move_direction_x_norm = (
        state["mouse_pos_x"] - SCREEN_WIDTH//2)/(SCREEN_WIDTH//2)
    move_direction_y_norm = (
        state["mouse_pos_y"] - SCREEN_HEIGHT//2)/(SCREEN_HEIGHT//2)
    attack = 1.0 if state["if_attack"] else 0.0
    defend = 1.0 if state["if_defend"] else 0.0
    yinyang = 1.0 if state["yinyang"] else 0.0
    pred_yinyang = 1.0 if state["pred_yinyang"] else 0.0
    degree = state["degree"]
    mob_features = []
    sorted_mobs = sorted(state["mob_pos"], key=lambda x: x["dist"])[
        :MAX_MOBS_NUM]
    for mob in sorted_mobs:
        x_norm = (mob["x_avg"] - SCREEN_WIDTH // 2) / (SCREEN_WIDTH // 2)
        y_norm = (mob["y_avg"] - SCREEN_HEIGHT // 2) / (SCREEN_HEIGHT // 2)
        width_norm = abs(mob["x_1"] - mob["x_2"]) / SCREEN_WIDTH
        height_norm = abs(mob["y_1"] - mob["y_2"]) / SCREEN_HEIGHT
        one_hot = one_hot_encode_mob(mob["name"])
        mob_features.append([x_norm, y_norm, width_norm, height_norm]+one_hot)
    padded_mob_features = mob_features + \
        [[0]*7 for _ in range(MAX_MOBS_NUM-len(mob_features))]
    dataset = {
        "state": {
            "health": health_norm,
            "degree": degree,
            "mobs": padded_mob_features,
            "yinyang": yinyang
        },
        "action": {
            "move_x": move_direction_x_norm,
            "move_y": move_direction_y_norm,
            "attack": attack,
            "defend": defend,
            "pred_yinyang": pred_yinyang
        }
    }
    return dataset


def log(event: str, type: str, show: bool = True):
    back_frame = sys._getframe().f_back
    if back_frame is not None:
        back_filename = os.path.basename(back_frame.f_code.co_filename)
        back_funcname = back_frame.f_code.co_name
        back_lineno = back_frame.f_lineno
    else:
        back_filename = "Unknown"
        back_funcname = "Unknown"
        back_lineno = "Unknown"
    now = datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    logger = f"[{time}] <{back_filename}:{back_lineno}> <{back_funcname}()> {type}: {event}"
    if type.lower() == "info":
        style = "green"
    elif type.lower() == "error":
        style = "red"
    elif type.lower() == "critical":
        style = "bold red"
    elif type.lower() == "event":
        style = "#ffab70"
    else:
        style = ""
    if show:
        console.print(logger, style=style)
    with open('latest.log', 'a', encoding='utf-8') as f:
        f.write(f'{logger}\n')


class FlorrDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with jsonlines.open(file_path) as reader:
            for item in reader:
                state = item["state"]
                action = item["action"]
                health = state["health"]
                degree = state["degree"]
                yinyang = state["yinyang"]
                mobs = [val for mob in state["mobs"]
                        for val in mob]
                state_tensor = torch.FloatTensor(
                    [health, degree, yinyang] + mobs)
                action_tensor = torch.FloatTensor([
                    action["move_x"],
                    action["move_y"],
                    action["attack"],
                    action["defend"],
                    action["pred_yinyang"]
                ])
                self.data.append((state_tensor, action_tensor))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class FlorrModel(nn.Module):
    def __init__(self, input_dim=73, output_dim=5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x[:, 0:2] = self.tanh(x[:, 0:2])  # Move output
        x[:, 2:5] = self.sigmoid(x[:, 2:5])  # Attack, Defend, YinYang output
        return x


def load_model(model_dir, device="cuda" if torch.cuda.is_available() else "cpu"):
    with open(model_dir + "/config.json", "r") as f:
        config = json.load(f)
    input_dim = config["input_dim"]
    output_dim = config["output_dim"]
    model = FlorrModel(input_dim, output_dim).to(device)
    model.load_state_dict(torch.load(
        model_dir + "/model.pth", map_location=device))
    model.eval()
    return model


def preprocess_state(state: dict) -> torch.Tensor:
    health = state["health"]
    degree = state["degree"]
    yinyang = state["yinyang"]
    mobs = [val for mob in state["mobs"] for val in mob]
    return torch.FloatTensor([health, degree, yinyang] + mobs)


def get_action(model, state):
    state_tensor = preprocess_state(state).unsqueeze(0).to("cuda")
    with torch.no_grad():
        action = model(state_tensor)
    return {
        "move_x": action[0, 0].item(),
        "move_y": action[0, 1].item(),
        "attack": 1 if action[0, 2] > 0.5 else 0,
        "defend": 1 if action[0, 3] > 0.5 else 0,
        "pred_yinyang": 1 if action[0, 4] > 0.5 else 0
    }


def get_if_equip(image, template, threshold=0.8):
    y = find_image(image, template, threshold)
    if y != None:
        y = y[0][0]
    if y == None:
        equipped = False
    elif 920 < y < 930:
        equipped = True
    else:
        equipped = False
    return equipped


def find_image(original, template, threshold=0.8):
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(original, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= threshold)
    if len(loc[0]) == 0:
        return None
    return loc

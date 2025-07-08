import time
from dataset_utils import *

if_attack = False
if_defend = False
mobs = []
(direction_x, direction_y) = (0, 0)
health = 0
degree = 0
toggle_yinyang = False


def mouse_click_thread():
    global if_attack, if_defend
    with pynput.mouse.Events() as event:
        for i in event:
            if isinstance(i, pynput.mouse.Events.Click):
                if i.button == pynput.mouse.Button.left:
                    if_attack = i.pressed
                if i.button == pynput.mouse.Button.right:
                    if_defend = i.pressed


def mouse_pos_thread():
    global direction_x, direction_y, health, toggle_yinyang
    yinyang_template = cv2.imread("./templates/yinyang.png")
    while True:
        pyautogui.keyDown("g")
        frame = pyautogui.screenshot(region=(0, 0, 1920, 1080))
        open_cv_image = np.array(frame)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
        toggle_yinyang = get_if_equip(open_cv_image, yinyang_template)
        direction_x, direction_y = pyautogui.position()
        health = check_health()["percent"]


def yolo_thread():
    global mobs
    mobs = []
    while True:
        frame = pyautogui.screenshot(region=(0, 0, 1920, 1080))
        open_cv_image = np.array(frame)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
        result = model.predict(frame, verbose=False)[0]
        names = result.names
        boxes = result.boxes
        things = boxes.data.tolist()
        mobs = []
        for method in things:
            confidence = method[4]
            if confidence < 0.5:
                continue
            new_method = []
            for i in method:
                new_method.append(round(i))
            x_1 = new_method[0]
            y_1 = new_method[1]
            x_2 = new_method[2]
            y_2 = new_method[3]
            x_avg = (x_1 + x_2) / 2
            y_avg = (y_1 + y_2) / 2
            dist = math.sqrt((x_avg - SCREEN_WIDTH//2)**2 +
                             (y_avg - SCREEN_HEIGHT//2)**2)
            name = new_method[5]
            object = names[name]
            mobs.append({"name": object, "x_1": x_1,
                         "y_1": y_1, "x_2": x_2, "y_2": y_2, "x_avg": x_avg, "y_avg": y_avg, "dist": dist})


def graph_thread():
    while True:
        frame = pyautogui.screenshot(region=(0, 0, 1920, 1080))
        open_cv_image = np.array(frame)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
        for mob in mobs:
            cv2.rectangle(open_cv_image, (mob["x_1"], mob["y_1"]),
                          (mob["x_2"], mob["y_2"]), (0, 255, 0), 2)
            cv2.putText(open_cv_image, mob["name"], (mob["x_1"], mob["y_1"] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.putText(open_cv_image, f"0: {if_attack}", (10, 1080-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.putText(open_cv_image, f"1: {if_defend}", (10, 1080-45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.putText(open_cv_image, f"[{direction_x}, {direction_y}]", (10, 1080-70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.putText(open_cv_image, f"health: {health}", (10, 1080-95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.putText(open_cv_image, f"degree: {degree}", (10, 1080-120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.putText(open_cv_image, f"yinyang: {toggle_yinyang}", (10, 1080-145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.imshow("Image", open_cv_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def save_thread():
    time_now = int(time.time())
    while True:
        if mobs != []:
            dict = {"health": health, "degree": degree, "yinyang": toggle_yinyang, "if_attack": if_attack,
                    "if_defend": if_defend, "mouse_pos_x": direction_x, "mouse_pos_y": direction_y, "mob_pos": mobs}
            dataset = state2dataset(dict)
            print(dataset)
            with open(f"./trains/data/{time_now}.jsonl", "a") as f:
                f.write(json.dumps(dataset)+"\n")
        time.sleep(0.1)


def rotation_server_thread():
    global degree
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
        global degree
        for item in content.textList:
            if item["size"] == 12:
                data = item
                canvas_mid = (data["canvasWidth"] // 2,
                              data["canvasHeight"] // 2)
                petal_point = (data["x"], data["y"])
                degree = math.atan2(
                    petal_point[1] - canvas_mid[1], petal_point[0] - canvas_mid[0]) / math.pi
        return {"status": "success", "received": data}
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000,
                log_level="critical", access_log=False)


if __name__ == "__main__":
    yolo_thread = threading.Thread(target=yolo_thread)
    yolo_thread.start()
    # graph_thread = threading.Thread(target=graph_thread)
    # graph_thread.start()
    mouse_click_thread = threading.Thread(target=mouse_click_thread)
    mouse_click_thread.start()
    mouse_pos_thread = threading.Thread(target=mouse_pos_thread)
    mouse_pos_thread.start()
    save_thread = threading.Thread(target=save_thread)
    save_thread.start()
    rotation_server_thread = threading.Thread(target=rotation_server_thread)
    rotation_server_thread.start()

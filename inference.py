from dataset_utils import *
from pynput import keyboard
import time

health = 0
mobs = []
action = {"move_x": 0, "move_y": 0, "attack": 0, "defend": 0}
mouse_pos_x = mouse_pos_y = 0
auto_suggest = False
yinyang = False
toggle_yinyang = False
degree = 0


def start_listener():
    with keyboard.Listener(on_press=key_press) as listener:
        listener.join()


def key_press(key):
    try:
        if key.char == 'p':
            global auto_suggest
            auto_suggest = not auto_suggest
    except AttributeError:
        pass


def yolo_thread():
    global mobs
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


def inference_thread():
    global action, health, mouse_pos_x, mouse_pos_y, pred_yinyang, mobs, degree
    inf_model = load_model("./models/auto_stf")
    while True:
        health = check_health()["percent"]
        dict = {"health": health, "if_attack": False,
                "if_defend": False, "mouse_pos_x": 1920//2, "mouse_pos_y": 1080//2, "mob_pos": mobs, "degree": degree, "yinyang": toggle_yinyang, "pred_yinyang": 0}
        state = state2dataset(dict)["state"]
        action = get_action(inf_model, state)
        print(action)
        mouse_pos_x = int(action["move_x"]*(1920//2) + (1920//2))
        mouse_pos_y = int(action["move_y"]*(1080//2) + (1080//2))


def graph_thread():
    global mobs
    while True:
        frame = pyautogui.screenshot(region=(0, 0, 1920, 1080))
        open_cv_image = np.array(frame)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
        for mob in mobs:
            cv2.rectangle(open_cv_image, (mob["x_1"], mob["y_1"]),
                          (mob["x_2"], mob["y_2"]), (0, 255, 0), 2)
            cv2.putText(open_cv_image, mob["name"], (mob["x_1"], mob["y_1"] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.putText(open_cv_image, f"0: {action['attack']}", (10, 1080-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.putText(open_cv_image, f"1: {action['defend']}", (10, 1080-45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.putText(open_cv_image, f"[{action['move_x']}, {action['move_y']}]", (10, 1080-70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.putText(open_cv_image, f"health: {health}", (10, 1080-95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.putText(open_cv_image, f"mobs: {len(mobs)}", (10, 1080-120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.putText(open_cv_image, f"auto_suggest: {auto_suggest}", (10, 1080-145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.putText(open_cv_image, f"degree: {degree}", (10, 1080-170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.putText(open_cv_image, f"yinyang: {yinyang}", (10, 1080-195),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.line(open_cv_image, (1920//2, 1080//2),
                 (mouse_pos_x, mouse_pos_y), (255, 0, 0), 2)
        cv2.imshow("Image", open_cv_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


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


def action_thread():
    global toggle_yinyang
    yinyang_template = cv2.imread("./templates/yinyang.png")
    pyautogui.keyDown("g")
    while True:
        if auto_suggest:
            open_cv_image = pyautogui.screenshot(region=(0, 0, 1920, 1080))
            open_cv_image = np.array(open_cv_image)
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
            toggle_yinyang = get_if_equip(open_cv_image, yinyang_template)
            if action["attack"] == 1:
                pyautogui.mouseDown(button='left')
            elif action["attack"] == 0:
                pyautogui.mouseUp(button='left')
            if action["defend"] == 1:
                pyautogui.mouseDown(button='right')
            elif action["defend"] == 0:
                pyautogui.mouseUp(button='right')
            if action["pred_yinyang"] == 1:
                if not toggle_yinyang:
                    pyautogui.write("2")
            elif action["pred_yinyang"] == 0:
                if toggle_yinyang:
                    pyautogui.write("2")
            pyautogui.moveTo(mouse_pos_x, mouse_pos_y, duration=0.1)
        else:
            time.sleep(0.5)


if __name__ == "__main__":
    yolo_thread = threading.Thread(target=yolo_thread)
    yolo_thread.start()
    inference_thread = threading.Thread(target=inference_thread)
    inference_thread.start()
    graph_thread = threading.Thread(target=graph_thread)
    graph_thread.start()
    action_thread = threading.Thread(target=action_thread)
    action_thread.start()
    listener_thread = threading.Thread(target=start_listener)
    listener_thread.start()
    rotation_server_thread = threading.Thread(target=rotation_server_thread)
    rotation_server_thread.start()

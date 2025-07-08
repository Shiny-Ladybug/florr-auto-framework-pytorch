import torch
from dataset_utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt


def custom_loss(pred, target, move_weight=0.5, action_weight=1.0):
    move_loss = nn.MSELoss()(pred[:, 0:2], target[:, 0:2])  # move_x 和 move_y
    attack_loss = nn.BCELoss()(pred[:, 2], target[:, 2])  # attack
    defend_loss = nn.BCELoss()(pred[:, 3], target[:, 3])  # defend
    yinyang_loss = nn.BCELoss()(pred[:, 4], target[:, 4])  # yinyang
    total_loss = move_weight * move_loss + action_weight * \
        (attack_loss + defend_loss + yinyang_loss)
    return total_loss, move_loss, attack_loss, defend_loss, yinyang_loss


def train(model, train_dataloader, val_dataloader, epochs=50, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    loss_history = []
    val_loss_history = []
    detailed_loss_history = {
        "move_loss": [], "attack_loss": [], "defend_loss": [], "yinyang_loss": []}
    val_detailed_loss_history = {
        "move_loss": [], "attack_loss": [], "defend_loss": [], "yinyang_loss": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        move_loss_total, attack_loss_total, defend_loss_total, yinyang_loss_total = 0.0, 0.0, 0.0, 0.0

        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for states, actions in pbar:
                states = states.to(device)
                actions = actions.to(device)
                pred = model(states)
                loss, move_loss, attack_loss, defend_loss, yinyang_loss = custom_loss(
                    pred, actions)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                move_loss_total += move_loss.item()
                attack_loss_total += attack_loss.item()
                defend_loss_total += defend_loss.item()
                yinyang_loss_total += yinyang_loss.item()

                pbar.set_postfix({
                    "Loss": loss.item(),
                    "Move": move_loss.item(),
                    "Attack": attack_loss.item(),
                    "Defend": defend_loss.item(),
                    "YinYang": yinyang_loss.item()
                })

        avg_loss = total_loss / len(train_dataloader)
        avg_move_loss = move_loss_total / len(train_dataloader)
        avg_attack_loss = attack_loss_total / len(train_dataloader)
        avg_defend_loss = defend_loss_total / len(train_dataloader)
        avg_yinyang_loss = yinyang_loss_total / len(train_dataloader)

        loss_history.append(avg_loss)
        detailed_loss_history["move_loss"].append(avg_move_loss)
        detailed_loss_history["attack_loss"].append(avg_attack_loss)
        detailed_loss_history["defend_loss"].append(avg_defend_loss)
        detailed_loss_history["yinyang_loss"].append(avg_yinyang_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_move_loss_total, val_attack_loss_total, val_defend_loss_total, val_yinyang_loss_total = 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            for states, actions in val_dataloader:
                states = states.to(device)
                actions = actions.to(device)
                pred = model(states)
                loss, move_loss, attack_loss, defend_loss, yinyang_loss = custom_loss(
                    pred, actions)

                val_loss += loss.item()
                val_move_loss_total += move_loss.item()
                val_attack_loss_total += attack_loss.item()
                val_defend_loss_total += defend_loss.item()
                val_yinyang_loss_total += yinyang_loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_move_loss = val_move_loss_total / len(val_dataloader)
        avg_val_attack_loss = val_attack_loss_total / len(val_dataloader)
        avg_val_defend_loss = val_defend_loss_total / len(val_dataloader)
        avg_val_yinyang_loss = val_yinyang_loss_total / len(val_dataloader)

        val_loss_history.append(avg_val_loss)
        val_detailed_loss_history["move_loss"].append(avg_val_move_loss)
        val_detailed_loss_history["attack_loss"].append(avg_val_attack_loss)
        val_detailed_loss_history["defend_loss"].append(avg_val_defend_loss)
        val_detailed_loss_history["yinyang_loss"].append(avg_val_yinyang_loss)

        scheduler.step(loss)

        print(
            f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, "
            f"Move Loss: {avg_move_loss:.4f}/{avg_val_move_loss:.4f}, "
            f"Attack Loss: {avg_attack_loss:.4f}/{avg_val_attack_loss:.4f}, "
            f"Defend Loss: {avg_defend_loss:.4f}/{avg_val_defend_loss:.4f}, "
            f"YinYang Loss: {avg_yinyang_loss:.4f}/{avg_val_yinyang_loss:.4f}"
        )

    return loss_history, val_loss_history, detailed_loss_history, val_detailed_loss_history


def load_dataset(path):
    if os.path.exists(path+"/config.json"):
        with open(path+"/config.json", "r") as f:
            config = json.load(f)
    model_name = config["model_name"]
    input_dim = config["input_dim"]
    output_dim = config["output_dim"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    dataset_path = config["dataset_path"]
    dataset_paths = os.listdir(dataset_path)
    full_dataset = []
    for path_ in dataset_paths:
        if path_.endswith(".jsonl"):
            with open(dataset_path+"\\"+path_, "r") as f:
                for line in f:
                    data = json.loads(line)
                    full_dataset.append(data)
    with open(path+"/dataset.jsonl", "w") as f:
        for data in full_dataset:
            f.write(json.dumps(data)+"\n")
    return model_name, path+"/dataset.jsonl", input_dim, output_dim, batch_size, epochs


if __name__ == "__main__":
    MODEL_NAME, FILE_PATH, INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, EPOCHS = load_dataset(
        "./trains")
    log(f"Training started with args: {FILE_PATH,INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, EPOCHS}", "INFO")
    dataset = FlorrDataset(FILE_PATH)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = FlorrModel(INPUT_DIM, OUTPUT_DIM)
    log(f"Model created: {model}", "INFO")
    loss_history, val_loss_history, detailed_loss_history, val_detailed_loss_history = train(
        model, train_dataloader, val_dataloader, epochs=EPOCHS)

    if not os.path.exists(f"./models/{MODEL_NAME}"):
        os.makedirs(f"./models/{MODEL_NAME}")
    log(f"Model saved to {MODEL_NAME}", "INFO")
    torch.save(model.state_dict(), f"./models/{MODEL_NAME}/model.pth")
    with open(f"./models/{MODEL_NAME}/config.json", "w") as f:
        json.dump({
            "model_name": MODEL_NAME,
            "input_dim": INPUT_DIM,
            "output_dim": OUTPUT_DIM,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS
        }, f)
    log("Model saved", "INFO")
    log("Training finished", "INFO")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), loss_history, label="Training Loss")
    plt.plot(range(1, EPOCHS + 1), val_loss_history, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.grid()
    plt.savefig(f"./models/{MODEL_NAME}/loss_curve.png")

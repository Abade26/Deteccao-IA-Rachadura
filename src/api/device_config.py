import yaml
import os

CONFIG_PATH = "configs/device_config.yaml"

def save_device_config(id, api_url):
    os.makedirs("configs", exist_ok=True)
    
    config = {
        "id": id,
        "api_url": api_url,
    }

    with open(CONFIG_PATH, "w") as file:
        yaml.dump(config, file)

    print(f"✅ Configuração salva em {CONFIG_PATH}")

def load_device_config():
    if not os.path.exists(CONFIG_PATH):
        return None
    with open(CONFIG_PATH, "r") as file:
        return yaml.safe_load(file)

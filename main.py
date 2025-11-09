from src.api.qr_reader import QRReader
from src.api.pair_client import PairClient
from src.api.device_config import save_device_config

def main():
    print("📷 Iniciando leitura do QR Code...")
    reader = QRReader()
    raw = reader.read_once()

    if not raw:
        print("Nenhum QR lido.")
        return

    payload = reader.parse_payload(raw)
    if not payload:
        print("QR inválido.")
        return

    client = PairClient(api_base_url=payload["api_url"])
    resposta = client.pair(id=payload["id"], codigo_pareador=payload["codigo_pareador"])

    if resposta is not None:
        save_device_config(payload["id"], payload["api_url"])
        print("✅ Dispositivo pareado e configuração salva!")
    else:
        print("❌ Pareamento falhou.")

if __name__ == "__main__":
    main()

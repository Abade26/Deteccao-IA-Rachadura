import requests

class PairClient:
    def __init__(self, api_base_url, timeout_sec=8):
        self.api_base_url = api_base_url.rstrip("/")
        self.timeout = timeout_sec

    def pair(self, id, codigo_pareador):
        url = f"{self.api_base_url}/device/pair"
        payload = {"id": id, "codigo_pareador": codigo_pareador}

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)

            print("📡 Enviando para:", url)
            print("📦 Payload:", payload)
            print("📥 Status code:", resp.status_code)
            print("📥 Resposta:", resp.text)

            if resp.status_code == 200:
                return resp.json() if resp.content else {}
            else:
                return None
        except Exception as e:
            print("❌ Erro de conexão com API:", e)
            return None

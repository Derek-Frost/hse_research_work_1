import requests

class LlamaAgent:
    def __init__(self, server_url, model_name):
        self.server_url = server_url
        self.model_name = model_name

    def send_prompt(self, prompt):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": 100,
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9
        }
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(self.server_url, json=payload, headers=headers)
            if response.status_code == 200:
                return response.json().get("choices", [{}])[0].get("text", "Ответ не получен")
            else:
                return f"Ошибка сервера: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Ошибка при выполнении запроса: {e}"

if __name__ == "__main__":
    agent = LlamaAgent(
        "http://localhost:8000/v1/completions",
        "nous-hermes-llama-2-7b.Q4_K_M.gguf"
    )
    while True:
        prompt = input("Введите промпт: ")
        if prompt.lower() in ["exit", "quit"]:
            break
        print("Ответ:", agent.send_prompt(prompt))

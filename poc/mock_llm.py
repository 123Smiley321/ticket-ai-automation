class LLMServiceError(Exception):
    pass


class MockLLMService:
    def __init__(self, is_active: bool = True):
        self.is_active = is_active

    def generate(self, prompt: str, context: str) -> str:
        if not self.is_active:
            raise LLMServiceError("LLM API unavailable")

        return f"Здравствуйте! Ответ по вашему запросу:\n{context}\nОбращайтесь, если появятся вопросы!"
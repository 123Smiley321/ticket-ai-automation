import re
import time
from typing import Tuple, List
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from poc.mock_llm import MockLLMService, LLMServiceError


class TicketMessage(BaseModel):
    ticket_id: str
    user_id: str
    text: str


class ProcessingOutput(BaseModel):
    ticket_id: str
    clean_text: str
    category: str
    confidence: float
    is_risky: bool
    status: str
    response: str
    queue: str
    elapsed_ms: float


KNOWLEDGE_BASE = {
    "payment": "Для привязки карты откройте Профиль -> Способы оплаты -> Добавить карту.",
    "subscription": "Управление подпиской доступно в Настройках аккаунта в разделе Подписки.",
    "refund_or_delete": "Заявка на возврат средств за списание и удаление профиля."
}


class SupportTicketPipeline:
    def __init__(self, llm: MockLLMService):
        self.llm = llm
        self.categories = list(KNOWLEDGE_BASE.keys())
        self.vectorizer = TfidfVectorizer()
        self.kb_vectors = self.vectorizer.fit_transform([KNOWLEDGE_BASE[c] for c in self.categories])

    def mask_pii(self, text: str) -> str:
        text = re.sub(r'\b\d{16}\b|\b(?:\d[ -]*?){16}\b', '[КАРТА]', text)
        text = re.sub(r'(\+7|8)\d{10}', '[ТЕЛЕФОН]', text)
        text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]', text)
        return text

    def classify_vector(self, text: str) -> Tuple[str, float, bool]:
        text_vector = self.vectorizer.transform([text])
        sims = cosine_similarity(text_vector, self.kb_vectors)[0]
        
        idx = sims.argmax()
        score = float(sims[idx])
        category = self.categories[idx]

        is_risky = category == "refund_or_delete" or any(
            w in text.lower() for w in ["вернуть", "возврат", "удал", "суд", "деньг"]
        )

        if is_risky:
            return "refund_or_delete", round(max(score, 0.40) + 0.50, 2), True

        if score < 0.12:
            return "unknown", 0.35, False

        return category, round(score + 0.50, 2), False

    def process(self, msg: TicketMessage) -> ProcessingOutput:
        start = time.time()

        clean_text = self.mask_pii(msg.text)
        category, confidence, is_risky = self.classify_vector(clean_text)

        if is_risky:
            try:
                draft = self.llm.generate(clean_text, "Проверить историю списаний вручную.")
            except LLMServiceError:
                draft = "Сбой LLM. Нужен ручной разбор."

            return ProcessingOutput(
                ticket_id=msg.ticket_id,
                clean_text=clean_text,
                category=category,
                confidence=confidence,
                is_risky=True,
                status="SUGGEST_TO_OPERATOR",
                response=f"[Черновик ответа]: {draft}",
                queue="senior_support",
                elapsed_ms=(time.time() - start) * 1000
            )

        if confidence < 0.70:
            return ProcessingOutput(
                ticket_id=msg.ticket_id,
                clean_text=clean_text,
                category=category,
                confidence=confidence,
                is_risky=False,
                status="ESCALATED_TO_HUMAN",
                response="Передано оператору связи с низкой уверенностью модели.",
                queue="general_support",
                elapsed_ms=(time.time() - start) * 1000
            )

        try:
            kb_doc = KNOWLEDGE_BASE.get(category, "Инструкция не найдена.")
            ans = self.llm.generate(clean_text, kb_doc)
            status = "AUTO_CLOSED"
            queue = "auto_closed_archive"
        except LLMServiceError:
            ans = "Ошибка сервиса генерации. Тикет перенаправлен оператору."
            status = "ESCALATED_TO_HUMAN"
            queue = "general_support"

        return ProcessingOutput(
            ticket_id=msg.ticket_id,
            clean_text=clean_text,
            category=category,
            confidence=confidence,
            is_risky=False,
            status=status,
            response=ans,
            queue=queue,
            elapsed_ms=(time.time() - start) * 1000
        )
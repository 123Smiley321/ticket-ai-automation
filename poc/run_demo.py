import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poc.pipeline import SupportTicketPipeline, TicketMessage
from poc.mock_llm import MockLLMService


def run_demo():
    print("=== Демонстрация работы PoC ===\n")

    normal_pipeline = SupportTicketPipeline(llm=MockLLMService(is_active=True))
    broken_pipeline = SupportTicketPipeline(llm=MockLLMService(is_active=False))

    t1 = TicketMessage(
        ticket_id="TCK-101",
        user_id="USR-01",
        text="Как привязать карту 4571123456789012 к аккаунту? Мой телефон +79991112233"
    )
    res1 = normal_pipeline.process(t1)
    print("1. Happy Path (Привязка карты):")
    print(f"   Текст без PII : {res1.clean_text}")
    print(f"   Категория     : {res1.category} (score: {res1.confidence})")
    print(f"   Статус        : {res1.status}")
    print(f"   Очередь       : {res1.queue}")
    print(f"   Ответ bot'а   :\n{res1.response}\n")

    t2 = TicketMessage(
        ticket_id="TCK-102",
        user_id="USR-02",
        text="Верните деньги за списание подписки и удалите профиль!"
    )
    res2 = normal_pipeline.process(t2)
    print("2. Risky Path (Возврат средств):")
    print(f"   Категория : {res2.category}, Рискованный: {res2.is_risky}")
    print(f"   Статус    : {res2.status}")
    print(f"   Очередь   : {res2.queue}")
    print(f"   Ответ     : {res2.response}\n")

    res3 = broken_pipeline.process(t1)
    print("3. Fallback Path (Сбой LLM):")
    print(f"   Статус  : {res3.status}")
    print(f"   Очередь : {res3.queue}")
    print(f"   Ответ   : {res3.response}\n")


if __name__ == "__main__":
    run_demo()
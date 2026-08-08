import pytest
from poc.pipeline import SupportTicketPipeline, TicketMessage
from poc.mock_llm import MockLLMService


def test_pii_cleaning():
    pipeline = SupportTicketPipeline(llm=MockLLMService())
    text = "Карта 4571123456789012, телефон +79990001122, email user@test.ru"
    cleaned = pipeline.mask_pii(text)
    assert "[КАРТА]" in cleaned
    assert "[ТЕЛЕФОН]" in cleaned
    assert "[EMAIL]" in cleaned
    assert "4571123456789012" not in cleaned


def test_happy_path():
    pipeline = SupportTicketPipeline(llm=MockLLMService())
    ticket = TicketMessage(ticket_id="1", user_id="1", text="Как привязать карту?")
    res = pipeline.process(ticket)
    assert res.status == "AUTO_CLOSED"
    assert res.category == "payment"


def test_risky_path():
    pipeline = SupportTicketPipeline(llm=MockLLMService())
    ticket = TicketMessage(ticket_id="2", user_id="2", text="Хочу вернуть деньги")
    res = pipeline.process(ticket)
    assert res.status == "SUGGEST_TO_OPERATOR"
    assert res.is_risky is True
    assert res.queue == "senior_support"


def test_llm_fallback():
    pipeline = SupportTicketPipeline(llm=MockLLMService(is_active=False))
    ticket = TicketMessage(ticket_id="3", user_id="3", text="Как привязать карту?")
    res = pipeline.process(ticket)
    assert res.status == "ESCALATED_TO_HUMAN"
    assert res.queue == "general_support"
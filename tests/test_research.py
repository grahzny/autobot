"""Tests for ResearchLedger -- notes, rationales, post-mortems."""

from __future__ import annotations

from autobot.research import (
    PostMortem,
    ResearchLedger,
    ResearchNote,
    TradeRationale,
)


class TestResearchNote:
    def test_auto_id(self):
        note = ResearchNote(ticker="BHP.AX", content="Test")
        assert note.id != ""

    def test_auto_timestamp(self):
        note = ResearchNote(ticker="BHP.AX", content="Test")
        assert note.timestamp > 0

    def test_to_dict(self):
        note = ResearchNote(ticker="BHP.AX", content="Undervalued", note_type="thesis")
        d = note.to_dict()
        assert d["ticker"] == "BHP.AX"
        assert d["note_type"] == "thesis"


class TestTradeRationale:
    def test_auto_id(self):
        r = TradeRationale(ticker="AAPL", thesis="Strong earnings")
        assert r.trade_id != ""

    def test_fields(self):
        r = TradeRationale(
            ticker="BHP.AX", direction="long",
            thesis="Iron ore demand", conviction=0.8,
            catalysts=["China stimulus"], risks=["Commodity downturn"],
            stop_loss=38.0, target=48.0, time_horizon="weeks",
        )
        d = r.to_dict()
        assert d["direction"] == "long"
        assert d["conviction"] == 0.8
        assert len(d["catalysts"]) == 1
        assert d["stop_loss"] == 38.0


class TestPostMortem:
    def test_fields(self):
        pm = PostMortem(
            trade_id="t1", ticker="AAPL",
            entry_price=150.0, exit_price=155.0, pnl=38.0,
            what_went_right=["Thesis confirmed"],
            what_went_wrong=["Sold too early"],
            lessons=["Hold winners longer"],
        )
        d = pm.to_dict()
        assert d["pnl"] == 38.0
        assert "Hold winners longer" in d["lessons"]


class TestResearchLedger:
    def test_add_note(self):
        ledger = ResearchLedger()
        note = ledger.add_note("BHP.AX", "Looks undervalued", note_type="thesis", conviction=0.7)
        assert len(ledger.notes) == 1
        assert note.conviction == 0.7

    def test_add_note_trims_at_200(self):
        ledger = ResearchLedger()
        for i in range(210):
            ledger.add_note("BHP.AX", f"Note {i}")
        assert len(ledger.notes) == 200

    def test_notes_for_ticker(self):
        ledger = ResearchLedger()
        ledger.add_note("BHP.AX", "Note 1")
        ledger.add_note("AAPL", "Note 2")
        ledger.add_note("BHP.AX", "Note 3")
        bhp_notes = ledger.notes_for_ticker("BHP.AX")
        assert len(bhp_notes) == 2
        assert all(n.ticker == "BHP.AX" for n in bhp_notes)

    def test_active_theses(self):
        ledger = ResearchLedger()
        ledger.add_note("BHP.AX", "Strong thesis", note_type="thesis", conviction=0.8)
        ledger.add_note("BHP.AX", "Weak thesis", note_type="thesis", conviction=0.4)
        ledger.add_note("AAPL", "Just an observation", note_type="observation", conviction=0.9)
        theses = ledger.active_theses()
        assert len(theses) == 1
        assert theses[0].content == "Strong thesis"

    def test_add_rationale(self):
        ledger = ResearchLedger()
        r = TradeRationale(ticker="BHP.AX", thesis="Iron ore play")
        ledger.add_rationale(r)
        assert len(ledger.rationales) == 1

    def test_rationale_for_trade(self):
        ledger = ResearchLedger()
        r = TradeRationale(trade_id="t1", ticker="BHP.AX", thesis="Test")
        ledger.add_rationale(r)
        found = ledger.rationale_for_trade("t1")
        assert found is not None
        assert found.ticker == "BHP.AX"

    def test_rationale_not_found(self):
        ledger = ResearchLedger()
        assert ledger.rationale_for_trade("nonexistent") is None

    def test_add_post_mortem(self):
        ledger = ResearchLedger()
        pm = PostMortem(trade_id="t1", ticker="BHP.AX", pnl=25.0, lessons=["Be patient"])
        ledger.add_post_mortem(pm)
        assert len(ledger.post_mortems) == 1

    def test_recent_lessons(self):
        ledger = ResearchLedger()
        pm1 = PostMortem(trade_id="t1", lessons=["Lesson 1", "Lesson 2"])
        pm2 = PostMortem(trade_id="t2", lessons=["Lesson 3"])
        ledger.add_post_mortem(pm1)
        ledger.add_post_mortem(pm2)
        lessons = ledger.recent_lessons(limit=2)
        assert len(lessons) == 2
        assert "Lesson 3" in lessons  # most recent first

    def test_weekly_review(self):
        ledger = ResearchLedger()
        ledger.add_weekly_review({"summary": "Good week"})
        assert len(ledger.weekly_reviews) == 1
        assert "timestamp" in ledger.weekly_reviews[0]

    def test_summary(self):
        ledger = ResearchLedger()
        ledger.add_note("BHP.AX", "Test", note_type="thesis", conviction=0.8)
        s = ledger.summary()
        assert s["total_notes"] == 1
        assert s["active_theses"] == 1

    def test_to_dict(self):
        ledger = ResearchLedger()
        ledger.add_note("BHP.AX", "Test")
        d = ledger.to_dict()
        assert len(d["notes"]) == 1

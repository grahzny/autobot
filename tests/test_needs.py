"""Tests for the needs/drives system."""

from __future__ import annotations

from autobot.needs import NeedsState, decay_needs, update_needs_from_events


class TestNeedsState:
    def test_initial_values(self):
        n = NeedsState()
        assert n.stimulation == 0.7
        assert n.meaning == 0.6
        assert n.belonging == 0.7
        assert n.competence == 0.7
        assert n.autonomy == 0.8

    def test_clamp(self):
        n = NeedsState(stimulation=1.5, meaning=-0.1)
        n.clamp()
        assert n.stimulation == 1.0
        assert n.meaning == 0.0

    def test_lowest_need(self):
        n = NeedsState(stimulation=0.1, meaning=0.5, belonging=0.9, competence=0.7, autonomy=0.8)
        name, val = n.lowest_need()
        assert name == "stimulation"
        assert val == 0.1

    def test_deficit_summary_no_deficits(self):
        n = NeedsState()
        assert n.deficit_summary() == "no critical deficits"

    def test_deficit_summary_with_deficits(self):
        n = NeedsState(stimulation=0.2, meaning=0.3)
        summary = n.deficit_summary()
        assert "stimulation" in summary
        assert "meaning" in summary

    def test_to_dict(self):
        n = NeedsState()
        d = n.to_dict()
        assert set(d.keys()) == {"stimulation", "meaning", "belonging", "competence", "autonomy"}


class TestDecayNeeds:
    def test_passive_decay(self):
        n = NeedsState()
        original_stim = n.stimulation
        decay_needs(n)
        assert n.stimulation < original_stim

    def test_decay_clamps_to_zero(self):
        n = NeedsState(stimulation=0.001)
        decay_needs(n)
        assert n.stimulation == 0.0

    def test_all_needs_decay(self):
        n = NeedsState()
        before = n.to_dict()
        decay_needs(n)
        after = n.to_dict()
        for key in before:
            assert after[key] < before[key], f"{key} did not decay"


class TestUpdateNeedsFromEvents:
    def test_new_encounter_restores_stimulation(self):
        n = NeedsState(stimulation=0.3)
        events = [{"type": "new_encounter"}]
        update_needs_from_events(n, events, 0, 0, 0)
        assert n.stimulation > 0.3

    def test_meaningful_exchange_restores_belonging(self):
        n = NeedsState(belonging=0.3)
        events = [{"type": "meaningful_exchange"}]
        update_needs_from_events(n, events, 0, 0, 0)
        assert n.belonging > 0.3

    def test_criticism_drains_competence(self):
        n = NeedsState(competence=0.7)
        events = [{"type": "criticism_received"}]
        update_needs_from_events(n, events, 0, 0, 0)
        assert n.competence < 0.7

    def test_self_reflection_restores_autonomy(self):
        n = NeedsState(autonomy=0.5)
        events = [{"type": "self_reflection"}]
        update_needs_from_events(n, events, 0, 0, 0)
        assert n.autonomy > 0.5

    def test_goal_stall_drains_meaning(self):
        n = NeedsState(meaning=0.6)
        update_needs_from_events(n, [], goal_stall_count=5, unanswered_proactive=0, seconds_since_interaction=0)
        assert n.meaning < 0.6

    def test_unanswered_proactive_drains_belonging(self):
        n = NeedsState(belonging=0.7)
        update_needs_from_events(n, [], goal_stall_count=0, unanswered_proactive=2, seconds_since_interaction=0)
        assert n.belonging < 0.7

    def test_long_silence_extra_drains_stimulation(self):
        n = NeedsState(stimulation=0.5)
        update_needs_from_events(n, [], goal_stall_count=0, unanswered_proactive=0, seconds_since_interaction=400)
        assert n.stimulation < 0.5

    def test_entity_spoke_drains_autonomy(self):
        n = NeedsState(autonomy=0.8)
        events = [{"type": "entity_spoke"}]
        update_needs_from_events(n, events, 0, 0, 0)
        assert n.autonomy < 0.8

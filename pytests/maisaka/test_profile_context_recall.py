from types import SimpleNamespace

import asyncio
import pytest

from src.maisaka.memory import person_profile
from src.services.memory_service import MemoryHit, MemorySearchResult


@pytest.mark.asyncio
async def test_multiple_complete_profiles_are_fetched_concurrently(monkeypatch):
    candidates = [person_profile.PersonProfileCandidate(person_id=str(i), person_name=f"用户{i}") for i in range(5)]
    person_profile._SCOPED_PROFILE_CACHE.clear()
    monkeypatch.setattr(person_profile, "collect_person_profile_candidates", lambda *args, **kwargs: candidates)
    monkeypatch.setattr(person_profile, "global_config", SimpleNamespace(a_memorix=SimpleNamespace(
        integration=SimpleNamespace(enable_person_profile_injection=True, person_profile_injection_max_profiles=5))))
    all_started = asyncio.Event()
    started = []
    async def query(name, **kwargs):
        started.append(kwargs['person_id'])
        assert kwargs['limit'] == 20 and kwargs['chat_id'] == 'current-group'
        assert not kwargs.get('cross_group', False)
        if len(started) == len(candidates):
            all_started.set()
        await asyncio.wait_for(all_started.wait(), timeout=1)
        return MemorySearchResult(hits=[MemoryHit(content='别名: 熟人\n' + '记忆' * 600 + '\n以前的约定')])
    monkeypatch.setattr(person_profile.memory_service, 'search', query)
    message = SimpleNamespace(session_id='current-group', message_info=SimpleNamespace(
        group_info=SimpleNamespace(group_id='10'), user_info=SimpleNamespace(user_id='20')))
    result = await person_profile.build_person_profile_injection_messages(anchor_message=message)
    assert len(started) == 5 and len(result) == 1
    assert result[0].count('别名: 熟人') == 5
    assert result[0].count('以前的约定') == 5
    assert '记忆' * 600 in result[0]


def test_recent_speakers_and_explicit_targets_are_prioritized(monkeypatch):
    monkeypatch.setattr(person_profile, '_sender_candidate', lambda msg, source: person_profile.PersonProfileCandidate(person_id=msg.message_id))
    def msg(mid):
        return SimpleNamespace(message_id=mid, raw_message=SimpleNamespace(components=[]),
                               message_info=SimpleNamespace(group_info=object()))
    result = person_profile.collect_person_profile_candidates(msg('current'), [msg(str(i)) for i in range(6)], max_profiles=5)
    assert [item.person_id for item in result] == ['current', '5', '4', '3', '2']

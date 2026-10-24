import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import longmemeval_g1_methods as g1
import longmemeval_u1_methods as m


def history():
    # Session 1 is listed first but dated last: list order is not time order.
    row = dict(question_id='q', haystack_session_ids=['a', 'b', 'c'],
               haystack_dates=['2023/05/20 (Sat) 10:00', '2023/05/01 (Mon) 09:00', '2023/05/10 (Wed) 09:00'],
               haystack_sessions=[[dict(role='user', content='I moved to Kyoto.'), dict(role='assistant', content='Nice.')],
                                  [dict(role='user', content='I live in Osaka.')],
                                  [dict(role='user', content='My cat is Tama.')]])
    return g1.turns(row)


def words(text):
    return len(text.split())


class SamplingTests(unittest.TestCase):
    def test_quota_seed_order_and_exclusion(self):
        meta = []
        for kind, n in m.QUOTA.items():
            for i in range(n + 2):
                qid = f'{kind}-{i}' + ('_abs' if kind == 'abstention' else '')
                meta.append({'question_id': qid, 'question_type': 'multi-session' if kind == 'abstention' else kind})
        first = m.sample(meta, set())
        self.assertEqual(len(first), sum(m.QUOTA.values()))
        self.assertEqual(first, m.sample(list(reversed(meta)), set()))
        dropped = first[0]
        second = m.sample(meta, {dropped})
        self.assertNotIn(dropped, second)
        self.assertEqual(sum(q.endswith('_abs') for q in second), m.QUOTA['abstention'])

    def test_short_stratum_refuses(self):
        with self.assertRaises(ValueError):
            m.sample([{'question_id': 'x', 'question_type': 'knowledge-update'}], set())


class WindowTests(unittest.TestCase):
    def test_g1_rule(self):
        self.assertEqual(m.window(19257, 1500, (16384, 32768)), 32768)
        self.assertEqual(m.window(13422, 120, (16384, 32768, 65536)), 16384)  # G1's measured rerank maximum
        self.assertIsNone(m.window(70000, 120, (16384, 32768, 65536)))


class TimeOrderTests(unittest.TestCase):
    def test_sessions_sorted_by_date_and_each_block_dated(self):
        ts = history()
        self.assertEqual([t['id'] for t in m.chronological(ts)], ['q/D2:1', 'q/D3:1', 'q/D1:1', 'q/D1:2'])
        text = m.render([ts[2], ts[1], ts[0]])  # rank order mixing sessions
        self.assertEqual(text.splitlines(), ['Date: 2023/05/01 (Mon) 09:00', '[q/D2:1] user: I live in Osaka.',
                                             'Date: 2023/05/20 (Sat) 10:00', '[q/D1:1] user: I moved to Kyoto.',
                                             '[q/D1:2] assistant: Nice.'])

    def test_recent_takes_latest_dates_not_list_end(self):
        ts = history()
        _, ids = m.recent(ts, 13, words)
        self.assertEqual(ids, ['q/D1:1', 'q/D1:2'])

    def test_memory_frozen_stop_rule_and_exclusion(self):
        ts = history(); lookup = {t['id']: t for t in ts}
        _, ids = m.memory(['q/D1:1', 'q/D2:1', 'q/D3:1'], lookup, ['q/D2:1'], 12, words)
        self.assertEqual(ids, ['q/D1:1'])  # D3 would exceed; stop, no skipping ahead

    def test_messages_carry_question_date_and_rule(self):
        chat = m.messages('c', 'r', 'Where do I live?', '2023/06/01 (Thu) 08:00')
        self.assertIn('most recent one is current', chat[0]['content'])
        self.assertTrue(chat[1]['content'].startswith('Current date: 2023/06/01'))
        self.assertTrue(chat[0]['content'].startswith(m.am.SYSTEM))


class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.ts = history()
        facts = {'q#2': [{'fact': 'User lives in Osaka.', 'turns': ['q/D2:1']}],
                 'q#3': [{'fact': 'User has a cat named Tama.', 'turns': ['q/D3:1']}],
                 'q#1': [{'fact': 'User moved to Kyoto.', 'turns': ['q/D1:1']}]}
        self.table = m.fact_table(self.ts, facts)
        self.vectors = {'q#2#0': [1., 0.], 'q#3#0': [0., 1.], 'q#1#0': [.9, .1]}

    def test_jobs_compare_only_against_earlier_sessions(self):
        jobs = m.update_jobs(self.table, self.vectors)
        self.assertEqual([j['session'] for j in jobs], ['q#3', 'q#1'])
        self.assertEqual(jobs[1]['earlier'][0], 'q#2#0')
        self.assertTrue(all(k != 'q#1#0' for j in jobs for k in j['earlier']))
        chat = m.update_messages(jobs[1], {f['key']: f for f in self.table})
        self.assertIn('[E1] (2023/05/01 (Mon) 09:00) User lives in Osaka.', chat[1]['content'])
        self.assertIn('[N1] User moved to Kyoto.', chat[1]['content'])

    def test_parse_keeps_valid_labels_only(self):
        job = {'earlier': ['q#2#0', 'q#3#0'], 'new': ['q#1#0']}
        text = '{"updates": [{"old": "E1", "new": "N1"}, {"old": "E9", "new": "N1"}, {"old": "N1", "new": "E1"}, {"old": "E1", "new": "N1"}]}'
        self.assertEqual(m.parse_updates(text, job), [{'old': 'q#2#0', 'new': 'q#1#0'}])
        self.assertIsNone(m.parse_updates('not json', job))
        self.assertEqual(m.parse_updates('{"updates": []}', job), [])

    def test_chain_follows_to_newest(self):
        facts = {'a': {'rank': 0}, 'b': {'rank': 1}, 'c': {'rank': 2}}
        links = [{'old': 'a', 'new': 'b'}, {'old': 'b', 'new': 'c'}, {'old': 'c', 'new': 'a'}]
        self.assertEqual(m.replacements(links[:2], facts), {'a': ['c'], 'b': ['c']})
        m.replacements(links, facts)  # a cycle terminates

    def test_versioned_inserts_update_and_note(self):
        links = [{'old': 'q#2#0', 'new': 'q#1#0'}]
        order, notes = m.versioned(['q/D2:1', 'q/D3:1'], self.table, links)
        self.assertEqual(order, ['q/D2:1', 'q/D1:1', 'q/D3:1'])
        self.assertEqual(notes, {'q/D2:1': ['Later update (2023/05/20 (Sat) 10:00): User moved to Kyoto.']})

    def test_payload_t_and_v(self):
        links = [{'old': 'q#2#0', 'new': 'q#1#0'}]
        raw = ['q/D2:1', 'q/D3:1']
        sorted_only = m.payload('S', self.ts, raw, words, self.table, links)
        t = m.payload('T', self.ts, raw, words, self.table, links)
        self.assertEqual(sorted_only, t)  # S and T differ only in the chat template
        v = m.payload('V', self.ts, raw, words, self.table, links)
        self.assertNotIn('Later update', t['retrieved'])
        self.assertEqual(t['notes_shown'], 0)
        self.assertEqual(v['recent_ids'], t['recent_ids'])
        with self.assertRaises(ValueError):
            m.payload('C-final-norw', self.ts, raw, words)

    def test_v_note_and_inserted_update_count_toward_budget(self):
        ts = history(); lookup = {t['id']: t for t in ts}
        order, notes = m.versioned(['q/D2:1'], self.table, [{'old': 'q#2#0', 'new': 'q#1#0'}])
        text, ids = m.memory(order, lookup, [], 100, words, notes)
        self.assertEqual(ids, ['q/D2:1', 'q/D1:1'])
        self.assertIn('  Later update (2023/05/20 (Sat) 10:00): User moved to Kyoto.', text)
        _, tight = m.memory(order, lookup, [], words(m.render([ts[2]], notes)), words, notes)
        self.assertEqual(tight, ['q/D2:1'])

if __name__ == '__main__':
    unittest.main()

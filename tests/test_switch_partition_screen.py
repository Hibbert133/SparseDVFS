import json
import math
import sqlite3
import tempfile
import unittest
from pathlib import Path

from rtx5090.switch_partition_screen import UUIDS, analyze, decode, pair_timing, sensitivity, sustained


class SwitchScreenTests(unittest.TestCase):
    def test_encoding_explicit_and_bounded(self):
        self.assertAlmostEqual(decode(-1768344239, 'hz', True), 2526.623057)
        for raw, unit, signed in [(-1768344239, 'hz', False), (2526623057, 'mhz', True),
                                  (float('nan'), 'hz', True), (-2**40, 'hz', True)]:
            with self.assertRaises(ValueError):
                decode(raw, unit, signed)

    def test_stability_does_not_bridge_gaps(self):
        self.assertIsNone(sustained([(0, 1), (20000000, 1)], bool, 10000000))
        data = [(i * 1000000, 1) for i in range(12)]
        self.assertEqual(sustained(data, bool, 10000000), dict(start_ns=0, confirmed_ns=10000000))

    def test_pair_common_origin_not_individual_max(self):
        cards = [dict(uuid=u, midpoint=dict(start_ns=t, confirmed_ns=t+2),
                      plateau=dict(start_ns=t, confirmed_ns=t+10))
                 for u, t in zip(UUIDS, (102, 110))]
        self.assertEqual(pair_timing(100, cards)['plateau_start_ms'], 10/1e6)
        cards[1]['plateau'] = None
        self.assertIsNone(pair_timing(100, cards)['plateau_start_ms'])

    def test_strict_eq3_and_no_budget_claim_without_input(self):
        model = dict(operators={'case': dict(latency_s=.005)})
        rows = sensitivity(model)
        self.assertEqual(rows[0]['repeated_same_shape_calls_for_strict_eq3']['case'], 6)
        self.assertIsNone(rows[0]['hypothetical_5pct_switch_count_ceiling'])
        self.assertEqual(sensitivity(model, 4)[1]['hypothetical_5pct_switch_count_ceiling'], 20)

    def test_sqlite_end_to_end_readonly_and_restoration_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture, stage = root/'capture', root/'stage'
            capture.mkdir(); stage.mkdir()
            report = dict(status='comparison_collected', restoration_verified=True,
                          control_backend='persistent_windows_bridge',
                          clock_anchors=[dict(wall_time_ns=0, perf_before_ns=0, perf_after_ns=0)],
                          transitions=[dict(begin_ns=1000000000, end_ns=6000000000, requested_mhz=2250)],
                          commands=[dict(uuid=u, option='-lgc', value='2250,2250', returncode=0,
                                         begin_ns=1000000000+i*5000000, command_ms=3)
                                    for i,u in enumerate(UUIDS)])
            (capture/'report.json').write_text(json.dumps(report))
            model = dict(schema='linear-latency-table-v1', identity_fingerprint='frozen',
                         operators={str(i): dict(latency_s=.0001) for i in range(8)})
            (stage/'stage-freeze.json').write_text(json.dumps(dict(
                identity_fingerprint='frozen',models=dict(linear_latency=model))))
            db = sqlite3.connect(capture/'gpu-clock.sqlite')
            db.executescript('''CREATE TABLE DIAGNOSTIC_EVENT(severity INTEGER);
                CREATE TABLE TARGET_INFO_SESSION_START_TIME(utcEpochNs INTEGER);
                INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES(0);
                CREATE TABLE TARGET_INFO_GPU(id INTEGER,uuid TEXT,busLocation TEXT);
                CREATE TABLE TARGET_INFO_GPU_METRICS(typeId INTEGER,metricId INTEGER,metricName TEXT);
                CREATE TABLE GPU_METRICS(timestamp INTEGER,typeId INTEGER,metricId INTEGER,value INTEGER);''')
            for i,u in enumerate(UUIDS):
                gid = 1-i  # Reversed Nsight ordering must not reverse ranks.
                db.execute('INSERT INTO TARGET_INFO_GPU VALUES(?,?,?)',(gid,u,'pci'+str(i)))
                db.execute('INSERT INTO TARGET_INFO_GPU_METRICS VALUES(?,?,?)',(gid,0,'GPC Clock Frequency [MHz]'))
                rows = []
                for ms in range(700, 6000):
                    hz = 2800000000 if ms < 1003+i*5 else 2200000000
                    rows.append((ms*1000000,gid,0,hz-2**32))
                db.executemany('INSERT INTO GPU_METRICS VALUES(?,?,?,?)',rows)
            db.commit(); db.close()
            result = analyze(capture,stage,'hz',True)
            self.assertEqual(result['transitions'][0]['pair']['plateau_start_ms'],8)
            self.assertEqual(result['transitions'][0]['pair']['plateau_confirmed_ms'],18)
            self.assertFalse(result['deployable'])
            self.assertFalse(result['capture_identity_binding_verified'])
            report['restoration_verified'] = False
            (capture/'report.json').write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                analyze(capture,stage,'hz',True)


if __name__ == '__main__':
    unittest.main()

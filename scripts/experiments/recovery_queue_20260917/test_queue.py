"""CPU-only safety tests for process identity and GPU handoff eligibility."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import wait_and_run as q


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.service=dict(pid=10,start_ticks='100',user='owner')
        self.watched=dict(pid=20,start_ticks='200',user='luorongchuan')
        self.m=dict(wait_for=[self.watched],shared_retriever=self.service,minimum_start_free_mib=32768)
        self.snap=dict(free_mib=60000,processes=[dict(pid=10,identity=self.service)])
    def check(self, processes, snap=None):
        with patch.object(q,'identity',side_effect=lambda pid:processes.get(pid)):
            return q.blockers(self.m,snap or self.snap)
    def test_wait_even_when_watched_temporarily_releases_gpu(self):
        self.assertIn('waiting_pid:20',self.check({10:self.service,20:self.watched}))
    def test_allow_existing_shared_retriever_after_exit(self):
        self.assertEqual([],self.check({10:self.service}))
    def test_other_gpu_job_blocks(self):
        s=dict(self.snap,processes=self.snap['processes']+[dict(pid=30,identity=dict(pid=30,start_ticks='300',user='someone'))])
        self.assertIn('gpu3_busy_pid:30',self.check({10:self.service},s))
    def test_restarted_retriever_requires_explicit_manifest_update(self):
        new=dict(self.service,start_ticks='999')
        self.assertIn('shared_retriever_identity_changed_or_missing',self.check({10:new}))
    def test_low_memory_blocks(self):
        self.assertIn('insufficient_free_memory',self.check({10:self.service},dict(self.snap,free_mib=32000)))
    def test_pid_reuse_is_not_original_job(self):
        self.assertFalse(q.same_process(dict(self.watched,start_ticks='999'),self.watched))
    def test_completion_requires_counts(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'status.json'; job={'expected_status':str(p)}
            self.assertFalse(q.complete(job))
            p.write_text('{"state":"complete"}')
            self.assertFalse(q.complete(job))
            p.write_text('{"state":"complete","completed":1,"total":2}')
            self.assertFalse(q.complete(job))
            p.write_text('{"state":"complete","completed":2,"total":2}')
            self.assertTrue(q.complete(job))


if __name__=='__main__': unittest.main()

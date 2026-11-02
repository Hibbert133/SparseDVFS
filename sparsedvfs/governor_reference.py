"""Pure event/state reference for section 4.3; no CUDA, threads or sysfs I/O.

Event readiness and host observation times are supplied by the caller. The
synthetic entry point uses virtual time, not an emulation of PLL hardware.
"""
from dataclasses import dataclass
from .paper_core import Frequency, finite


@dataclass
class Ticket:
    id: int
    block_id: int
    run: Frequency
    upcoming: Frequency | None
    estimated_ns: int
    lead_ns: int
    want_lookahead: bool
    submitted: bool = False
    boundary_entered: bool = False
    start_seen_ns: int | None = None
    due_ns: int | None = None
    issued: bool = False
    ended: bool = False


class GovernorReference:
    def __init__(self, legal_opps, initial, mode='lookahead'):
        if mode not in ('serial', 'lookahead'):
            raise ValueError('Reference mode must be serial or lookahead')
        self.opps = {Frequency.of(f) for f in legal_opps}
        self.initial = Frequency.of(initial)
        if self.initial not in self.opps:
            raise ValueError('Initial reference vector must belong to the supplied grid')
        self.mode, self.cpu_owner, self.gpu_owner = mode, 0, 0
        self.cpu_requested = self.initial.cpu_khz
        self.resources_requested = self.initial.values[1:]
        self.sequence, self.closed, self.tickets, self.events = 0, False, {}, []

    def _record(self, kind, now_ns, ticket=None, **extra):
        if isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns < 0:
            raise ValueError('Host observation time must be a nonnegative integer in ns')
        self.events.append(dict(kind=kind, host_ns=now_ns, ticket=ticket.id if ticket else None,
                                cpu_requested_khz=self.cpu_requested,
                                gpu_emc_requested=list(self.resources_requested), **extra))

    def _ticket(self, ticket_id):
        if ticket_id not in self.tickets:
            raise ValueError('Unknown reference ticket')
        return self.tickets[ticket_id]

    def begin_submission(self, block, upcoming, boost_khz, switch_us, now_ns):
        if self.closed:
            raise RuntimeError('Reference governor is closed')
        run = Frequency.of(block['triplet'])
        nxt = Frequency.of(upcoming['triplet']) if upcoming else None
        if run not in self.opps or (nxt is not None and nxt not in self.opps):
            raise ValueError('Block vector is outside the supplied OPP grid')
        if boost_khz not in {f.cpu_khz for f in self.opps} or boost_khz < run.cpu_khz:
            raise ValueError('CPU submission boost must be a legal rate >= run rate')
        estimate = round(finite(block['estimated_ms'], 'block estimate', positive=True) * 1e6)
        lead = round(finite(switch_us, 'switch observation', positive=True) * 1000)
        want = bool(self.mode == 'lookahead' and nxt and block['amortized_strict_gt'])
        if want and estimate <= lead:
            raise ValueError('Look-ahead estimate must exceed the lead time')
        self.sequence += 1
        ticket = Ticket(self.sequence, block['block_id'], run, nxt, estimate, lead, want)
        self.tickets[ticket.id] = ticket
        self.cpu_owner, self.cpu_requested = ticket.id, boost_khz
        self._record('cpu_submission_boost', now_ns, ticket)
        return ticket.id

    def end_submission(self, ticket_id, now_ns):
        ticket = self._ticket(ticket_id)
        if self.closed or ticket.submitted:
            raise RuntimeError('Closed controller or duplicate submission end')
        ticket.submitted = True
        if ticket.id != self.cpu_owner:
            self._record('cpu_submission_release_stale', now_ns, ticket)
            return 'stale'
        self.cpu_requested = ticket.run.cpu_khz
        self._record('cpu_submission_release', now_ns, ticket)
        return 'released'

    def gpu_boundary(self, ticket_id, now_ns):
        ticket = self._ticket(ticket_id)
        if self.closed or ticket.id <= self.gpu_owner:
            raise RuntimeError('GPU boundaries must follow submission order on the open main stream')
        self.gpu_owner = ticket.id
        ticket.boundary_entered = True
        self.resources_requested = ticket.run.values[1:]
        self._record('gpu_stream_boundary', now_ns, ticket)

    def observe_start(self, ticket_id, now_ns):
        ticket = self._ticket(ticket_id)
        if self.closed or not ticket.boundary_entered or ticket.start_seen_ns is not None:
            raise RuntimeError('Start must be observed once after its stream boundary')
        ticket.start_seen_ns = now_ns
        ticket.due_ns = now_ns + max(ticket.estimated_ns - ticket.lead_ns, 0)
        self._record('gpu_start_observed', now_ns, ticket, due_ns=ticket.due_ns)

    def poll(self, ticket_id, now_ns, end_ready, *, after_write_end_ready=False):
        ticket = self._ticket(ticket_id)
        if self.closed:
            return 'closed'
        if ticket.ended:
            return 'complete'
        if ticket.start_seen_ns is None or not ticket.submitted:
            return 'waiting'
        if now_ns < ticket.start_seen_ns:
            raise ValueError('Poll predates the observed start')
        if end_ready:
            if ticket.want_lookahead and not ticket.issued:
                self._record('lookahead_missed_end_ready', now_ns, ticket)
            ticket.ended = True
            self._record('gpu_end_observed', now_ns, ticket)
            return 'complete'
        if not ticket.want_lookahead or ticket.issued or now_ns < ticket.due_ns:
            return 'waiting'
        ticket.issued = True
        if ticket.id != self.gpu_owner:
            self._record('lookahead_skipped_stale', now_ns, ticket)
            return 'stale'
        target = ticket.upcoming.values[1:]
        if target == self.resources_requested:
            self._record('lookahead_noop_same_target', now_ns, ticket)
            return 'no_change'
        self.resources_requested = target
        kind = ('lookahead_written_end_complete_postcheck' if after_write_end_ready
                else 'lookahead_written_end_pending')
        self._record(kind, now_ns, ticket)
        return 'ambiguous_after_write' if after_write_end_ready else 'request_issued_end_pending'

    def restore(self, now_ns):
        if not self.closed:
            self.cpu_requested = self.initial.cpu_khz
            self.resources_requested = self.initial.values[1:]
            self.closed = True
            self._record('virtual_snapshot_restored', now_ns)
        return dict(closed=self.closed, actual_hardware_restored=False,
                    source='virtual reference state only')

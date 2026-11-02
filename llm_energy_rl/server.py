from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from .workload import Request


@dataclass
class WindowStats:

    duration: float = 0.0
    busy_time: float = 0.0  # prefill + decode time
    stall_time: float = 0.0  # prefill time that held up requests already decoding
    arrivals: int = 0
    tokens: int = 0  # output tokens produced
    prefill_tokens: int = 0
    decode_steps: int = 0
    decode_time: float = 0.0
    ttft: list[float] = field(default_factory=list)  # requests that got their first token
    finished: list[Request] = field(default_factory=list)
    queue_len: int = 0
    active: int = 0
    oldest_wait: float = 0.0  # age of the longest-waiting queued request

    @property
    def inter_token_s(self) -> float:
        return (self.decode_time + self.stall_time) / self.decode_steps if self.decode_steps else 0.0


class Server:
    def __init__(self, backend, max_batch: int):
        self.backend = backend
        self.max_batch = max_batch
        self.reset([], 0.0)

    def reset(self, trace: list[Request], t0: float) -> None:
        self.trace = trace
        self.t0 = t0
        self.next = 0  # index of the next request that has not arrived yet
        self.queue: deque[Request] = deque()
        self.active: list[Request] = []
        self.done: list[Request] = []

    @property
    def all_done(self) -> bool:
        return self.next >= len(self.trace) and not self.queue and not self.active

    def run_until(self, t_end: float) -> WindowStats:
        b, st = self.backend, WindowStats()
        t_start = b.now()
        while True:
            now = b.now()
            while self.next < len(self.trace) and self.t0 + self.trace[self.next].arrival <= now:
                self.queue.append(self.trace[self.next])
                self.next += 1
                st.arrivals += 1
            if now >= t_end:
                break
            free = self.max_batch - len(self.active)
            if free > 0 and self.queue:  # admit: prefill produces each request's first token
                group = [self.queue.popleft() for _ in range(min(free, len(self.queue)))]
                had_active = bool(self.active)
                b.prefill(group)
                t = b.now()
                for r in group:
                    r.first_token_t, r.n_out = t, 1
                    st.ttft.append(t - (self.t0 + r.arrival))
                self.active.extend(group)
                st.prefill_tokens += sum(r.prompt_len for r in group)
                st.tokens += len(group)
                st.busy_time += t - now
                if had_active:
                    st.stall_time += t - now
                self._retire(st, t)
                continue
            next_arrival = self.t0 + self.trace[self.next].arrival if self.next < len(self.trace) else math.inf
            if self.active: 
                k = min(r.out_len - r.n_out for r in self.active)
                until = min(t_end, next_arrival) if free > 0 else t_end
                steps, elapsed = b.decode(k, until)
                for r in self.active:
                    r.n_out += steps
                st.tokens += steps * len(self.active)
                st.decode_steps += steps
                st.decode_time += elapsed
                st.busy_time += elapsed
                self._retire(st, b.now())
                continue
            b.idle_until(min(t_end, next_arrival))
        now = b.now()
        st.duration = now - t_start
        st.queue_len, st.active = len(self.queue), len(self.active)
        st.oldest_wait = now - (self.t0 + self.queue[0].arrival) if self.queue else 0.0
        return st

    def _retire(self, st: WindowStats, t: float) -> None:
        rows = [i for i, r in enumerate(self.active) if r.n_out >= r.out_len]
        if not rows:
            return
        for i in rows:
            self.active[i].finish_t = t
            st.finished.append(self.active[i])
            self.done.append(self.active[i])
        self.backend.remove(rows)
        gone = set(rows)
        self.active = [r for i, r in enumerate(self.active) if i not in gone]

"""Read-only monitoring and terminal display for parallel mini trajectories."""
from threading import Lock
import shutil
import sys
import time


class TrajectoryStates:
    def __init__(self):
        self.lock = Lock()
        self.entries = {}

    def set(self, index, phase, *, agent=None, max_steps=0):
        now = time.monotonic()
        with self.lock:
            started = self.entries.get(index, {}).get('started', now)
            self.entries[index] = dict(phase=phase, agent=agent, max_steps=max_steps,
                                       started=started, since=now, signature=None)

    def remove(self, index):
        with self.lock:
            self.entries.pop(index, None)

    def snapshot(self):
        now = time.monotonic()
        rows = []
        with self.lock:
            for index, entry in sorted(self.entries.items()):
                phase, round_number = entry['phase'], 0
                agent = entry['agent']
                if agent is not None:
                    # Observe the native agent; never wrap execute/query or mutate it.
                    round_number = agent.n_calls
                    tail = agent.messages[-1:]
                    role = tail[0].get('role') if tail else None
                    phase = 'command' if role == 'assistant' else 'model'
                    if role == 'exit':
                        phase = 'finalizing'
                signature = (phase, round_number)
                if entry['signature'] is not None and signature != entry['signature']:
                    entry['since'] = now
                entry['signature'] = signature
                rows.append((index + 1, phase, round_number, entry['max_steps'],
                             int(now - entry['since']), int(now - entry['started'])))
        return rows


def duration(seconds):
    return f'{seconds // 60:02d}:{seconds % 60:02d}'


class CollectionProgress:
    """Main-thread rendering; workers only update the shared status registry."""
    def __init__(self, total, stats, states=None):
        self.total, self.stats = total, stats
        self.states = states or TrajectoryStates()
        self.stream = sys.stderr
        self.tty = self.stream.isatty()
        self.started = time.monotonic()
        self.last = None
        self.last_print = self.started
        self.lines = 0

    def __enter__(self):
        self.update(0)
        return self

    def clear(self):
        if self.tty and self.lines:
            # Each rendered line is ASCII and clipped to avoid terminal wrapping.
            self.stream.write(f'\x1b[{self.lines}A\r\x1b[J')
            self.lines = 0

    def update(self, active):
        done = self.stats['records_attempted']
        saved = self.stats['trajectories_written']
        steps = self.stats['interaction_steps']
        stopping = bool(self.stats['stopped_on_error'])
        rows = self.states.snapshot()
        state = (done, saved, steps, active, stopping, tuple(row[:4] for row in rows))
        now = time.monotonic()
        # Redirected logs report changes and a heartbeat during long waits.
        if not self.tty and state == self.last and now - self.last_print < 30:
            return
        self.last, self.last_print = state, now
        filled = min(20, 20 * done // self.total)
        lines = [f"[{'#' * filled}{'-' * (20 - filled)}] {done}/{self.total} "
                 f"({100 * done / self.total:.0f}%) | saved={saved} steps={steps} "
                 f"active={active} elapsed={duration(int(now - self.started))}"
                 + (' | stopping' if stopping else '')]
        for number, phase, round_number, maximum, phase_time, total_time in rows:
            step = f' round={round_number}/{maximum}' if maximum else ''
            lines.append(f'  #{number}{step} {phase} | phase={duration(phase_time)} total={duration(total_time)}')
        self.clear()
        if self.tty:
            width = max(1, shutil.get_terminal_size(fallback=(120, 24)).columns - 1)
            lines = [line[:width] for line in lines]
        print('\n'.join(lines), file=self.stream, flush=True)
        self.lines = len(lines) if self.tty else 0

    def log(self, message):
        self.clear()
        print(message, file=self.stream, flush=True)

    def __exit__(self, *_):
        self.stream.flush()

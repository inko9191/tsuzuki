"""macOS resource observations for the registered long-context experiment."""
from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
import time

import locomo_answer_audit_measure as audit

GIB = 1024**3


def parse_os(vm, swap, pressure):
    match = re.search(r'page size of (\d+) bytes', vm)
    if not match:
        raise RuntimeError('vm_stat page size unavailable')
    page = int(match[1])
    fields = {k.strip(): int(v) for k, v in re.findall(r'^([^:\n]+):\s+(\d+)\.', vm, re.M)}
    required = ('Pages free', 'Pages inactive', 'Pages speculative', 'Swapins', 'Swapouts')
    if not all(k in fields for k in required):
        raise RuntimeError('vm_stat resource fields unavailable')
    used = re.search(r'used\s*=\s*([0-9.]+)([KMGT])', swap)
    if not used:
        raise RuntimeError('swap usage unavailable')
    return {'page_size': page, 'available_bytes_estimate': page * sum(fields[k] for k in required[:3]),
        'available_definition': 'free+inactive+speculative pages; reclaimable estimate, no purgeable double count',
        'swapins': fields['Swapins'], 'swapouts': fields['Swapouts'],
        'swap_used_bytes': float(used[1]) * 1024**('KMGT'.index(used[2])+1),
        'pressure_level': int(pressure.strip()), 'monotonic_seconds': time.monotonic()}


def observe_os():
    vm = subprocess.check_output(['/usr/bin/vm_stat'], text=True)
    swap = subprocess.check_output(['/usr/sbin/sysctl', '-n', 'vm.swapusage'], text=True)
    pressure = subprocess.check_output(['/usr/sbin/sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], text=True)
    return parse_os(vm, swap, pressure)


def violation(memory, system):
    if memory['physical_footprint_bytes'] > 48*GIB:
        return 'Qwen physical footprint above48GiB'
    if system['available_bytes_estimate'] < 8*GIB:
        return 'available memory estimate below8GiB'
    if system['pressure_level'] >= 4:
        return 'macOS memory pressure critical'
    return None


def swap_increased(initial, current):
    return any(current[k] > initial[k] for k in ('swapins', 'swapouts', 'swap_used_bytes'))


class Watch:
    """50ms Qwen peak sampling and1s OS observations; no Qwen requests."""
    def __init__(self, pid, notify):
        self.pid, self.notify = pid, notify
        self.initial = observe_os()
        self.peak = audit.memory(pid)
        issue = violation(self.peak, self.initial)
        if issue:
            raise RuntimeError(issue)
        self.system = [self.initial]
        self.samples = 0
        self.swap_affected = False
        self.error = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def loop(self):
        next_os = time.monotonic()+1
        try:
            while not self.stop_event.wait(.05):
                current = audit.memory(self.pid)
                self.samples += 1
                self.peak = {k:max(v,self.peak[k]) for k,v in current.items()}
                system = self.system[-1]
                if time.monotonic() >= next_os:
                    system = observe_os()
                    self.system.append(system)
                    next_os = time.monotonic()+1
                    if not self.swap_affected and swap_increased(self.initial, system):
                        self.swap_affected = True
                        self.notify('swap increase', {'initial':self.initial, 'observed':system,
                            'time_interpretation':'reference only; swap affected'})
                issue = violation(current, system)
                if issue:
                    raise RuntimeError(issue)
        except BaseException as error:
            self.error = f'{type(error).__name__}: {error}'
            self.notify('resource monitoring failure', {'error':self.error, 'peak':self.peak})
            os.kill(os.getpid(), signal.SIGINT)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join(timeout=3)

    def record(self):
        return {'peak':self.peak, 'samples50ms':self.samples, 'OS_samples1s':self.system,
            'initial_OS':self.initial, 'swap_affected':self.swap_affected, 'monitor_error':self.error,
            'time_interpretation':'reference only; swap affected' if self.swap_affected else
                'no observed increment in swap; pre-existing swap separately recorded'}

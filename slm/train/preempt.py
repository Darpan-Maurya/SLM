"""Turning a kill signal into a clean checkpoint."""
from __future__ import annotations

import os
import signal
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class PreemptionGuard:
    """Reasons a run should stop early, unified behind ``should_stop()``."""

    max_runtime_sec: int = 0
    poll_cloud: bool = False
    poll_interval: float = 5.0
    stop_file: str = ""

    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _closed: threading.Event = field(default_factory=threading.Event, init=False)
    _reason: str = field(default="", init=False)
    _start: float = field(default_factory=time.monotonic, init=False)
    _prev: dict = field(default_factory=dict, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)

    SIGNALS = ("SIGTERM", "SIGINT", "SIGUSR1", "SIGQUIT", "SIGHUP")

    def install(self) -> PreemptionGuard:
        for name in self.SIGNALS:
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                self._prev[sig] = signal.signal(sig, self._handle)
            except (ValueError, OSError):
                pass          # not the main thread, or not supported here
        self._closed.clear()
        if self.poll_cloud or self.stop_file:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def uninstall(self) -> None:
        """Restore signal handlers and shut the watcher thread down."""
        self._closed.set()
        for sig, prev in self._prev.items():
            try:
                signal.signal(sig, prev)
            except (ValueError, OSError):
                pass
        self._prev.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # -- signal handling ----------------------------------------------------- #
    def _handle(self, signum, _frame) -> None:
        name = signal.Signals(signum).name
        if self._stop.is_set():
            # Second signal: the operator means it. Do not try to be clever.
            print(f"\n[preempt] {name} again - exiting immediately", flush=True)
            os._exit(130)
        self._reason = f"signal:{name}"
        self._stop.set()
        print(
            f"\n[preempt] caught {name}: finishing the current step, saving, "
            f"then exiting (send it again to abort now)",
            flush=True,
        )

    # -- background conditions ----------------------------------------------- #
    def _poll(self) -> None:
        while not self._stop.is_set() and not self._closed.is_set():
            if self.stop_file and os.path.exists(self.stop_file):
                self.trigger("stop-file")
                return
            if self.poll_cloud and self._cloud_termination():
                self.trigger("cloud-preemption-notice")
                return
            # wait on the event, not sleep, so shutdown is immediate
            self._closed.wait(self.poll_interval)

    @staticmethod
    def _cloud_termination() -> bool:
        probes = [
            # AWS IMDSv2 needs a token first; IMDSv1 still answers on many AMIs
            ("http://169.254.169.254/latest/meta-data/spot/instance-action", {}),
            ("http://metadata.google.internal/computeMetadata/v1/instance/preempted",
             {"Metadata-Flavor": "Google"}),
            ("http://169.254.169.254/metadata/scheduledevents?api-version=2020-07-01",
             {"Metadata": "true"}),
        ]
        for url, headers in probes:
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=1.0) as resp:
                    body = resp.read(256).decode("utf-8", "ignore").strip().upper()
                    if resp.status == 200 and body and body not in ("FALSE", "{}"):
                        if "EVENTS" in body and '"EVENTS":[]' in body.replace(" ", ""):
                            continue
                        return True
            except (urllib.error.URLError, OSError, ValueError, TimeoutError):
                continue
        return False

    # -- public API ---------------------------------------------------------- #
    def trigger(self, reason: str) -> None:
        self._reason = reason
        self._stop.set()
        print(f"\n[preempt] {reason}: will checkpoint and exit", flush=True)

    def should_stop(self) -> bool:
        if self._stop.is_set():
            return True
        if self.max_runtime_sec and self.elapsed > self.max_runtime_sec:
            self.trigger(f"max_runtime {self.max_runtime_sec}s reached")
            return True
        return False

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def __enter__(self) -> PreemptionGuard:
        return self.install()

    def __exit__(self, *exc) -> None:
        self.uninstall()

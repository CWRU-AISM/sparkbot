"""
USB lifecycle-safety tests: serialization, reset bounding/backoff,
the reset kill-switch (SPARK_ALLOW_USB_RESET, default OFF), and
shutdown join-before-close.

Context: the host hard-crashes only while the SPARK server is live, with a
kernel fingerprint of unclaimed-usbfs control traffic on the two Kinects at
every lifecycle transition plus xHCI TRB-ring corruption. These tests pin
the process-level invariants that keep USB lifecycle churn bounded and
serialized. All SDK objects are mocked -- zero hardware contact.
"""

import logging
import threading
import time

import pytest

from spark_real.perception import camera_base, kinect, realsense
import spark_real.pipeline as pipeline_mod
from spark_real.pipeline import SPARKRealPipeline


# Fake pyrealsense2 surface


class _FakeRSDevice:
    def __init__(self, serial="123", usb_type="3.2"):
        self._serial = serial
        self._usb = usb_type
        self.reset_calls = 0

    def get_info(self, key):
        if key == "serial_number":
            return self._serial
        return self._usb

    def hardware_reset(self):
        self.reset_calls += 1


class _FakeRS:
    """Counts context creations; serves one fake device."""

    class camera_info:
        serial_number = "serial_number"
        usb_type_descriptor = "usb_type_descriptor"

    def __init__(self):
        self.context_calls = 0
        self.device = _FakeRSDevice()

    def context(self):
        self.context_calls += 1
        outer = self

        class _Ctx:
            def query_devices(self):
                return [outer.device]

        return _Ctx()


class _FakePipe:
    def __init__(self):
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1


def _make_rs_cam(monkeypatch, fake_rs):
    monkeypatch.setattr(realsense, "HAS_REALSENSE", True)
    monkeypatch.setattr(realsense, "rs", fake_rs)
    monkeypatch.setattr(realsense, "_RS_CONTEXT", None, raising=False)
    monkeypatch.delenv("SPARK_WRIST_COLOR_ONLY", raising=False)
    monkeypatch.delenv("SPARK_NO_USB_RESET", raising=False)
    # Reset paths are OFF by default now; the recovery tests below are about
    # what happens once an operator arms them.
    monkeypatch.setenv("SPARK_ALLOW_USB_RESET", "1")
    c = realsense.RealSenseCamera(serial="123")
    c._usb2_mode = False
    c._pipeline = _FakePipe()
    # Model an opened, streaming camera (open() sets this); recovery is
    # only ever attempted on a camera that was successfully opened.
    c._running = True
    return c


@pytest.fixture(autouse=True)
def _reset_process_state():
    """Recovery attempts are bounded per process; reset between tests."""
    realsense.RealSenseCamera._recover_attempts_total = 0
    realsense.RealSenseCamera._last_recover_ts = None
    yield
    realsense.RealSenseCamera._recover_attempts_total = 0
    realsense.RealSenseCamera._last_recover_ts = None


# 1. Kill-switch: zero reset traffic, log-and-degrade


def test_kill_switch_disables_hardware_reset(monkeypatch, caplog):
    fake_rs = _FakeRS()
    c = _make_rs_cam(monkeypatch, fake_rs)
    monkeypatch.setenv("SPARK_NO_USB_RESET", "1")
    with caplog.at_level(logging.WARNING):
        assert c._recovery_allowed() is False
        assert c._try_recover() is False
    # No USB traffic of any kind: no enumeration, no reset, no pipeline churn.
    assert fake_rs.context_calls == 0
    assert fake_rs.device.reset_calls == 0
    assert c._pipeline.stop_calls == 0
    assert any("reset paths disabled" in r.getMessage() for r in caplog.records)


# 2. Reset bounding + exponential backoff (max attempts per process lifetime)


def test_recovery_bounded_with_exponential_backoff(monkeypatch):
    fake_rs = _FakeRS()
    c = _make_rs_cam(monkeypatch, fake_rs)
    c._start_pipeline = lambda: None
    # Do not actually sleep the 6s re-enumeration wait.
    monkeypatch.setattr(c._closing, "wait", lambda *a, **k: False)
    now = [1000.0]
    monkeypatch.setattr(camera_base._time, "monotonic", lambda: now[0])

    assert realsense.RealSenseCamera.MAX_RECOVER_ATTEMPTS == 3

    # Attempt 1 allowed immediately.
    assert c._recovery_allowed() is True
    assert c._try_recover() is True
    assert fake_rs.device.reset_calls == 1

    # Backoff gap 1: 30 s.
    now[0] += 5.0
    assert c._recovery_allowed() is False
    now[0] += 26.0  # t + 31
    assert c._recovery_allowed() is True
    assert c._try_recover() is True
    assert fake_rs.device.reset_calls == 2

    # Backoff gap 2: 60 s.
    now[0] += 31.0
    assert c._recovery_allowed() is False
    now[0] += 30.0  # t + 61
    assert c._recovery_allowed() is True
    assert c._try_recover() is True
    assert fake_rs.device.reset_calls == 3

    # Attempts exhausted for the rest of the process lifetime.
    now[0] += 1e6
    assert c._recovery_allowed() is False
    assert c._try_recover() is False
    assert fake_rs.device.reset_calls == 3


# 3. Lifecycle serialization: recovery cannot reset while another camera is
#    mid-open/close (both funnel through USB_LIFECYCLE_LOCK)


def test_recovery_serialized_by_lifecycle_lock(monkeypatch):
    fake_rs = _FakeRS()
    c = _make_rs_cam(monkeypatch, fake_rs)
    c._start_pipeline = lambda: None
    monkeypatch.setattr(c._closing, "wait", lambda *a, **k: False)

    started = threading.Event()
    done = threading.Event()

    def _worker():
        started.set()
        c._try_recover()
        done.set()

    assert camera_base.USB_LIFECYCLE_LOCK.acquire(timeout=2.0)
    try:
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        assert started.wait(2.0)
        time.sleep(0.3)
        # Blocked on the lifecycle lock: no reset traffic yet.
        assert fake_rs.device.reset_calls == 0
        assert not done.is_set()
    finally:
        camera_base.USB_LIFECYCLE_LOCK.release()
    assert done.wait(2.0)
    assert fake_rs.device.reset_calls == 1


# 4. RealSense shutdown: reader thread joined BEFORE pipeline.stop()


def test_realsense_close_joins_thread_before_stop(monkeypatch):
    fake_rs = _FakeRS()
    c = _make_rs_cam(monkeypatch, fake_rs)
    order = []

    t = threading.Thread(target=lambda: c._closing.wait(5.0), daemon=True)

    class _OrderedPipe:
        def stop(self):
            order.append(("stop", t.is_alive()))

    c._pipeline = _OrderedPipe()
    t.start()
    c._thread = t
    c.close()
    assert order == [("stop", False)]


def test_realsense_close_skips_stop_when_thread_stuck(monkeypatch, caplog):
    fake_rs = _FakeRS()
    c = _make_rs_cam(monkeypatch, fake_rs)
    monkeypatch.setattr(realsense.RealSenseCamera, "CLOSE_JOIN_TIMEOUT_S", 0.2)
    stuck = threading.Event()
    t = threading.Thread(target=stuck.wait, args=(10.0,), daemon=True)
    t.start()
    c._thread = t
    pipe = _FakePipe()
    c._pipeline = pipe
    with caplog.at_level(logging.WARNING):
        c.close()
    # Never stop the pipeline out from under a live reader thread.
    assert pipe.stop_calls == 0
    stuck.set()


# 5. Kinect shutdown: capture thread joined BEFORE device.stop()


def test_kinect_close_joins_capture_thread_before_device_stop():
    k = kinect.AzureKinectCamera(device_id=0)
    order = []

    t_holder = {}

    class _Dev:
        def stop(self):
            order.append(("stop", t_holder["t"].is_alive()))

    k._device = _Dev()

    def _loop():
        while not k._stop_flag:
            time.sleep(0.01)

    t = threading.Thread(target=_loop, daemon=True)
    t_holder["t"] = t
    t.start()
    k._thread = t
    k.close()
    assert order == [("stop", False)]


def test_kinect_close_skips_device_stop_when_thread_stuck(monkeypatch, caplog):
    monkeypatch.setattr(
        kinect.AzureKinectCamera, "JOIN_TIMEOUT_S", 0.1, raising=False
    )
    monkeypatch.setattr(kinect.AzureKinectCamera, "JOIN_GRACE_S", 0.1, raising=False)
    k = kinect.AzureKinectCamera(device_id=0)
    calls = []

    class _Dev:
        def stop(self):
            calls.append("stop")

    k._device = _Dev()
    stuck = threading.Event()
    t = threading.Thread(target=stuck.wait, args=(10.0,), daemon=True)
    t.start()
    k._thread = t
    with caplog.at_level(logging.WARNING):
        k.close()
    # A capture thread that will not die means libk4a is wedged; stopping the
    # device from another thread races its in-flight get_capture.
    assert calls == []
    stuck.set()


# 6. Enumerate/probe once per process


def test_realsense_probe_enumerates_once(monkeypatch):
    fake = _FakeRS()
    monkeypatch.setattr(pipeline_mod, "rs", fake)
    monkeypatch.setattr(pipeline_mod, "_REALSENSE_PROBE_RESULT", None, raising=False)
    assert pipeline_mod._realsense_device_available() is True
    assert pipeline_mod._realsense_device_available() is True
    assert pipeline_mod._realsense_device_available() is True
    assert fake.context_calls == 1


# 7. Pipeline shutdown ordering: quiesce RealSense recovery machinery FIRST,
#    so a hardware_reset can never fire while a Kinect is mid-close.


def test_shutdown_quiesces_realsense_recovery_before_closing_kinects():
    p = object.__new__(SPARKRealPipeline)
    order = []

    class _K:
        def close(self):
            order.append("kinect_close")

    class _R:
        def request_stop(self):
            order.append("rs_request_stop")

        def close(self):
            order.append("rs_close")

    p._kinect = _K()
    p._kinect2 = None
    p._realsense = _R()
    p._robot = None
    p._initialized = True
    p.shutdown()
    assert order == ["rs_request_stop", "kinect_close", "rs_close"]


# 9. close() reports failure, stays retryable, and gates the reset path.
#
# The old close() set self._closed = True BEFORE joining the capture thread,
# then returned early if the thread would not die. Result: a failed close was
# indistinguishable from a successful one, could never be retried (the
# reentrancy guard was already tripped), and silently satisfied
# /api/kinect_reset's "handles released" precondition -- so uhubctl cut Vbus
# on a hub whose device still had a live capture thread doing libusb I/O.


def _stuck_kinect(monkeypatch, stuck):
    monkeypatch.setattr(
        kinect.AzureKinectCamera, "JOIN_TIMEOUT_S", 0.05, raising=False
    )
    monkeypatch.setattr(kinect.AzureKinectCamera, "JOIN_GRACE_S", 0.05, raising=False)
    k = kinect.AzureKinectCamera(device_id=0)

    class _Dev:
        def stop(self):
            pass

    k._device = _Dev()
    t = threading.Thread(target=stuck.wait, args=(10.0,), daemon=True)
    t.start()
    k._thread = t
    return k


def test_close_returns_false_when_capture_thread_will_not_exit(monkeypatch):
    stuck = threading.Event()
    k = _stuck_kinect(monkeypatch, stuck)
    try:
        assert k.close() is False
        # And it must stay retryable: _closed must NOT be latched by a
        # failed attempt.
        assert getattr(k, "_closed", False) is False
        assert k.close() is False
    finally:
        stuck.set()


def test_close_succeeds_on_retry_once_the_thread_exits(monkeypatch):
    stuck = threading.Event()
    k = _stuck_kinect(monkeypatch, stuck)
    assert k.close() is False
    stuck.set()
    k._thread.join(timeout=2.0)
    assert k.close() is True
    assert k._closed is True
    # A completed close short-circuits and still reports success.
    assert k.close() is True


def test_close_returns_true_on_a_clean_release():
    k = kinect.AzureKinectCamera(device_id=0)
    calls = []

    class _Dev:
        def stop(self):
            calls.append("stop")

    k._device = _Dev()
    k._thread = None
    assert k.close() is True
    assert calls == ["stop"]
    assert camera_base.teardown_in_flight() is False


# 10. Teardown-in-flight: USB_LIFECYCLE_LOCK does not cover an abandoned
#     libk4a stop(), so a separate flag has to.


def test_teardown_flag_outlives_the_abandoned_release_wait(monkeypatch):
    monkeypatch.setattr(
        kinect.AzureKinectCamera, "RELEASE_JOIN_TIMEOUT_S", 0.1, raising=False
    )
    k = kinect.AzureKinectCamera(device_id=0)
    hang = threading.Event()

    class _HangingDev:
        def stop(self):
            hang.wait(10.0)

    k._device = _HangingDev()
    k._thread = None
    try:
        assert k.close() is False  # abandoned wait is not a clean release
        # The lifecycle lock is free again...
        assert camera_base.USB_LIFECYCLE_LOCK.acquire(blocking=False)
        camera_base.USB_LIFECYCLE_LOCK.release()
        # ...but libk4a is still on the bus, and the flag says so.
        assert camera_base.teardown_in_flight() is True
        assert camera_base.await_teardown_clear(timeout=0.2) is False
    finally:
        hang.set()
    assert camera_base.await_teardown_clear(timeout=5.0) is True
    assert camera_base.teardown_in_flight() is False


def test_realsense_recovery_refuses_while_a_teardown_is_in_flight(monkeypatch):
    fake_rs = _FakeRS()
    c = _make_rs_cam(monkeypatch, fake_rs)
    camera_base._teardown_begin()
    try:
        assert c._recovery_allowed() is False
        assert c._try_recover() is False
        assert fake_rs.device.reset_calls == 0
    finally:
        camera_base._teardown_end()
    assert c._recovery_allowed() is True


def test_kinect_open_refuses_while_a_teardown_is_in_flight(monkeypatch):
    monkeypatch.setattr(kinect, "HAS_K4A", True)
    k = kinect.AzureKinectCamera(device_id=0)
    camera_base._teardown_begin()
    try:
        monkeypatch.setattr(kinect, "await_teardown_clear", lambda *a, **kw: False)
        with pytest.raises(RuntimeError, match="teardown is still in flight"):
            k.open()
    finally:
        camera_base._teardown_end()


# 11. The kill-switch is inverted: reset paths are OFF unless armed.


def test_usb_reset_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SPARK_NO_USB_RESET", raising=False)
    monkeypatch.delenv("SPARK_ALLOW_USB_RESET", raising=False)
    assert camera_base.usb_reset_disabled() is True


def test_usb_reset_can_be_armed_explicitly(monkeypatch):
    monkeypatch.delenv("SPARK_NO_USB_RESET", raising=False)
    monkeypatch.setenv("SPARK_ALLOW_USB_RESET", "1")
    assert camera_base.usb_reset_disabled() is False


def test_legacy_no_reset_still_wins_over_the_arm_flag(monkeypatch):
    monkeypatch.setenv("SPARK_ALLOW_USB_RESET", "1")
    monkeypatch.setenv("SPARK_NO_USB_RESET", "1")
    assert camera_base.usb_reset_disabled() is True


def test_capture_loop_device_restart_is_off_by_default(monkeypatch):
    """
    The bounded stop/start in _capture_loop re-runs SET_INTERFACE bandwidth
    negotiation while the OTHER Kinect's isochronous reservation is live. It
    used to fire unattended after 40 consecutive failures.
    """
    monkeypatch.delenv("SPARK_NO_USB_RESET", raising=False)
    monkeypatch.delenv("SPARK_ALLOW_USB_RESET", raising=False)

    class _TE(Exception):
        pass

    monkeypatch.setattr(kinect, "K4ATimeoutException", _TE, raising=False)
    k = kinect.AzureKinectCamera(device_id=0)
    k._lock = threading.Lock()
    k._stop_flag = False
    calls = []
    seen = threading.Event()

    class _BrokenDev:
        def get_capture(self, timeout=None):
            calls.append("get")
            if len(calls) > 200:
                seen.set()
            raise RuntimeError("bus is angry")

        def stop(self):
            calls.append("stop")

        def start(self):
            calls.append("start")

    k._device = _BrokenDev()
    monkeypatch.setattr(camera_base._time, "sleep", lambda s: None)
    t = threading.Thread(target=k._capture_loop, daemon=True)
    t.start()
    assert seen.wait(5.0)
    k._stop_flag = True
    t.join(timeout=5.0)
    # Well past _FAIL_RESTART_THRESHOLD=40 and no lifecycle traffic at all.
    assert "stop" not in calls and "start" not in calls

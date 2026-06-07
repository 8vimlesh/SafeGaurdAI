"""Audio alert utilities for PPE Safety System.

Provides a simple cross-platform alarm. Prefers `winsound` on Windows,
falls back to `pyaudio` if available, otherwise no-op with a printed message.
"""
from pathlib import Path
import sys
import time

try:
    import winsound
    _HAS_WINSOUND = True
except Exception:
    _HAS_WINSOUND = False

try:
    import pyaudio
    import numpy as np
    _HAS_PYAUDIO = True
except Exception:
    _HAS_PYAUDIO = False

def is_critical_violation(class_name: str) -> bool:
    """Return True for classes considered critical violations.

    Currently 'without_mask' is considered critical.
    """
    return str(class_name).lower() == "without_mask"


def play_alarm(duration=3):
    """Play an alarm beep for `duration` seconds.

    Uses `winsound.Beep` on Windows. If unavailable, tries to emit a tone
    using `pyaudio`. If neither is available, falls back to a console bell.
    """
    try:
        secs = float(duration)
    except Exception:
        secs = 3.0

    # Use winsound on Windows
    if _HAS_WINSOUND:
        try:
            # winsound.Beep takes frequency (Hz) and duration (ms)
            from safeguardai.config import ALARM_FREQUENCY, ALARM_DURATION
            freq = int(ALARM_FREQUENCY)
            dur_ms = int(ALARM_DURATION) if ALARM_DURATION else int(secs * 1000)
            winsound.Beep(freq, dur_ms)
            return
        except Exception:
            # Try alternative Windows system sounds as a fallback
            try:
                try:
                    winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
                except Exception:
                    winsound.MessageBeep()
                return
            except Exception:
                pass

    # Try pyaudio if available
    if _HAS_PYAUDIO:
        try:
            from safeguardai.config import ALARM_FREQUENCY
            p = pyaudio.PyAudio()
            fs = 44100
            duration_secs = secs
            f = int(ALARM_FREQUENCY)
            samples = (np.sin(2*np.pi*np.arange(fs*duration_secs)*f/fs)).astype(np.float32)
            stream = p.open(format=pyaudio.paFloat32, channels=1, rate=fs, output=True)
            stream.write(samples.tobytes())
            stream.stop_stream()
            stream.close()
            p.terminate()
            return
        except Exception:
            pass

    # As a last resort, try system bell or a short pause
    try:
        print('\a', end='')
    except Exception:
        pass
    time.sleep(secs)

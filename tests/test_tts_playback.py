import threading
import time

import numpy as np
import pytest

from evil_jarvis.tts.playback import AudioPlayer, resample


class FakeStream:
    """Pulls audio through the player's callback the way PortAudio would."""

    latency = 0.0

    def __init__(self, player, block=512):
        self.player, self.block, self.played = player, block, []
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.running:
            out = np.empty((self.block, 1), dtype=np.float32)
            self.player._on_audio(out, self.block, None, None)
            self.played.append(out[:, 0].copy())
            time.sleep(0.001)

    def close(self):
        self.running = False


@pytest.fixture
def player():
    player = AudioPlayer()
    player._stream = FakeStream(player)
    yield player
    player.close()


def test_play_waits_until_the_audio_is_out(player):
    player.play(np.full(5000, 0.5, dtype=np.float32), player.rate)
    played = np.concatenate(player._stream.played)
    assert int(np.count_nonzero(played == 0.5)) == 5000


def test_silence_between_replies(player):
    time.sleep(0.02)
    assert all(not chunk.any() for chunk in player._stream.played)


def test_stream_is_reused(player):
    stream = player._stream
    for _ in range(3):
        player.play(np.ones(100, dtype=np.float32) * 0.1, player.rate)
    assert player._stream is stream


def test_resample_keeps_duration():
    samples = np.sin(np.linspace(0, 50, 16_000)).astype(np.float32)
    assert len(resample(samples, 16000, 22050)) == 22050
    assert resample(samples, 22050, 22050) is samples

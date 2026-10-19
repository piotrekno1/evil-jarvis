import threading
import time
import unittest

import numpy as np

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


def player_with_fake_stream():
    player = AudioPlayer()
    player._stream = FakeStream(player)
    return player


class PlaybackTests(unittest.TestCase):
    def test_play_waits_until_the_audio_is_out(self):
        player = player_with_fake_stream()
        tone = np.full(5000, 0.5, dtype=np.float32)
        player.play(tone, player.rate)
        played = np.concatenate(player._stream.played)
        self.assertEqual(int(np.count_nonzero(played == 0.5)), 5000)
        player.close()

    def test_silence_between_replies(self):
        player = player_with_fake_stream()
        time.sleep(0.02)
        self.assertTrue(all(not chunk.any() for chunk in player._stream.played))
        player.close()

    def test_stream_is_reused(self):
        player = player_with_fake_stream()
        stream = player._stream
        for _ in range(3):
            player.play(np.ones(100, dtype=np.float32) * 0.1, player.rate)
        self.assertIs(player._stream, stream)
        player.close()

    def test_resample_keeps_duration(self):
        samples = np.sin(np.linspace(0, 50, 16_000)).astype(np.float32)
        self.assertEqual(len(resample(samples, 16_000, 22_050)), 22_050)
        self.assertIs(resample(samples, 22_050, 22_050), samples)


if __name__ == "__main__":
    unittest.main()

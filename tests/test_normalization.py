import unittest

from backend.jejenorm import load_dataset, normalize_text


class NormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = load_dataset()

    def assert_normalizes_to(self, text, expected):
        normalized, _ = normalize_text(text, self.rules)
        self.assertEqual(normalized, expected)

    def test_galit_sample_normalizes_xa_and_inyo(self):
        self.assert_normalizes_to(
            "g4L1T 4K0 s4 iNyO!!! pWeD3 xA kAyA hNdI kA pUmUNtA dIt??",
            "galit ako sa inyo! pwede siya kaya hindi ka pumunta dito?",
        )

    def test_greeting_sample(self):
        self.assert_normalizes_to(
            "H3y u!!! kamuzta nA?",
            "hey you! kamusta na?",
        )

    def test_love_sample(self):
        self.assert_normalizes_to(
            "lOvE u 4eVeR!!!",
            "love you forever!",
        )

    def test_lyfe_normalizes_to_life(self):
        self.assert_normalizes_to(
            "H3y g0rl frnD!! k0y4 nNG AkO s4 lyfe st4y C0Ol 4lWaYz P4L!!",
            "hey girl friend! kaya ang ako sa life tayo cool always pa!",
        )


if __name__ == "__main__":
    unittest.main()

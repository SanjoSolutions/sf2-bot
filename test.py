import unittest

from detect_animation import detect_ken_shoryuken


class Test(unittest.TestCase):
    def test_ken_shoryuken_detection(self):
        frame_results = detect_ken_shoryuken()

        self.assertIn({'name': 'shoryuken', 'frame': 1, 'sprite_number': 0}, frame_results[0])
        self.assertIn({'name': 'shoryuken', 'frame': 2, 'sprite_number': 1}, frame_results[1])
        self.assertIn({'name': 'shoryuken', 'frame': 3, 'sprite_number': 2}, frame_results[2])
        self.assertIn({'name': 'shoryuken', 'frame': 4, 'sprite_number': 3}, frame_results[3])
        self.assertIn({'name': 'shoryuken', 'frame': 5, 'sprite_number': 4}, frame_results[4])
        self.assertIn({'name': 'shoryuken', 'frame': 6, 'sprite_number': 5}, frame_results[5])


if __name__ == '__main__':
    unittest.main()

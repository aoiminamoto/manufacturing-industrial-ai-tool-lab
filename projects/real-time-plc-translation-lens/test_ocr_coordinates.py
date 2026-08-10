import unittest

from ocr_coordinates import build_regions


class OcrCoordinateTests(unittest.TestCase):
    def test_normalizes_pixel_boxes_and_orders_by_position(self):
        regions = build_regions(
            texts=["下段運転", "Upper English", "上段停止"],
            scores=[0.9, 0.99, 0.8],
            boxes=[[500, 500, 750, 550], [0, 0, 100, 20], [100, 100, 300, 150]],
            width=1000,
            height=1000,
        )
        self.assertEqual([region["jp"] for region in regions], ["上段停止", "下段運転"])
        self.assertEqual(regions[0]["bbox"], [100, 100, 300, 150])
        self.assertEqual([region["id"] for region in regions], [1, 2])

    def test_filters_low_confidence_and_invalid_boxes(self):
        regions = build_regions(
            texts=["低信頼", "無効枠", "正常運転"],
            scores=[0.2, 0.9, 0.9],
            boxes=[[0, 0, 20, 20], [30, 30, 30, 40], [50, 60, 150, 100]],
            width=200,
            height=200,
        )
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0]["bbox"], [250, 300, 750, 500])

    def test_rejects_nonpositive_image_dimensions(self):
        with self.assertRaises(ValueError):
            build_regions([], [], [], width=0, height=100)


if __name__ == "__main__":
    unittest.main()

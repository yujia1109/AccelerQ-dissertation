from __future__ import annotations

from pathlib import Path
import sys
import unittest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from operator_selection import select_precise_gradient_index  # noqa: E402


class OperatorSelectionTests(unittest.TestCase):
    def test_empty_candidates_are_a_terminal_state(self) -> None:
        self.assertEqual(
            select_precise_gradient_index({}, previous_index=2, check_duplicate=True),
            (None, None),
        )

    def test_largest_absolute_gradient_is_selected(self) -> None:
        selected, ignored = select_precise_gradient_index(
            {0: 0.2, 1: -0.7, 2: 0.4},
            previous_index=None,
            check_duplicate=True,
        )
        self.assertEqual(selected, 1)
        self.assertIsNone(ignored)

    def test_duplicate_largest_uses_second_and_ignores_first(self) -> None:
        selected, ignored = select_precise_gradient_index(
            {0: 0.2, 1: -0.7, 2: 0.4},
            previous_index=1,
            check_duplicate=True,
        )
        self.assertEqual(selected, 2)
        self.assertEqual(ignored, 1)

    def test_single_duplicate_remains_selectable(self) -> None:
        self.assertEqual(
            select_precise_gradient_index(
                {3: -0.5}, previous_index=3, check_duplicate=True
            ),
            (3, None),
        )


if __name__ == "__main__":
    unittest.main()

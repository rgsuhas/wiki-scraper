import unittest
from unittest.mock import patch, MagicMock

class Calculator:
    def add(self, a, b):
        return a + b
    
    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

class TestCalculator(unittest.TestCase):
    """Test suite for Calculator class."""
    
    def setUp(self):
        """Run before each test."""
        self.calc = Calculator()
    
    def test_add_positive_numbers(self):
        """Test addition with positive numbers."""
        self.assertEqual(self.calc.add(2, 3), 5)
    
    def test_divide_by_zero_raises_error(self):
        """Test that dividing by zero raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.calc.divide(10, 0)
        self.assertIn("Cannot divide by zero", str(context.exception))
    
    @patch('requests.get')
    def test_api_call_mocking(self, mock_get):
        """Test with mocked external dependency."""
        mock_get.return_value.json.return_value = {'result': 42}
        # Test code using requests.get
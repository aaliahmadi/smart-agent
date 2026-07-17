import pytest
from unittest.mock import patch
from app.utils import has_internet

class TestHasInternet:
    
    @patch('app.utils.requests.get')
    def test_has_internet_true(self, mock_get):
        """Test when internet is available"""
        mock_get.return_value.status_code = 200
        
        result = has_internet()
        assert result is True
        mock_get.assert_called_once_with("https://www.google.com", timeout=3)
    
    @patch('app.utils.requests.get')
    def test_has_internet_false_timeout(self, mock_get):
        """Test when request times out (no internet)"""
        import requests
        mock_get.side_effect = requests.exceptions.Timeout
        
        result = has_internet()
        assert result is False
    
    @patch('app.utils.requests.get')
    def test_has_internet_false_connection_error(self, mock_get):
        """Test when connection fails"""
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError
        
        result = has_internet()
        assert result is False
    
    @patch('app.utils.requests.get')
    def test_has_internet_false_generic_exception(self, mock_get):
        """Test when any other exception occurs"""
        mock_get.side_effect = Exception("Something went wrong")
        
        result = has_internet()
        assert result is False

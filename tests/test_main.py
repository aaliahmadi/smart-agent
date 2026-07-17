import pytest
from streamlit.testing.v1 import AppTest

class TestMainApp:
    
    def test_app_initialization(self):
        """Test that the Streamlit app loads without errors"""
        at = AppTest.from_file("app/main.py")
        at.run()
        
        assert at.session_state is not None
        assert at.title[0].value == "🤖 Smart AI Agent (Gemma3 + Web Tools)"
    
    def test_clear_chat_button_exists(self):
        """Test that the clear chat button appears"""
        at = AppTest.from_file("app/main.py")
        at.run()
        
        buttons = [b for b in at.button if "Clear Chat History" in b.label]
        assert len(buttons) == 1
    
    def test_chat_input_exists(self):
        """Test that the chat input field exists"""
        at = AppTest.from_file("app/main.py")
        at.run()
        
        assert len(at.chat_input) == 1
        assert at.chat_input[0].label == "Ask me anything..."

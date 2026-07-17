import pytest
from unittest.mock import patch, MagicMock
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from app.agent_logic import run_agent

class TestRunAgent:
    
    def setup_method(self):
        """Create a fresh chat history for each test"""
        self.chat_history = StreamlitChatMessageHistory()
        self.chat_history.clear()
    
    @patch('app.agent_logic.has_internet')
    @patch('app.agent_logic.ChatOllama')
    @patch('app.agent_logic.load_tools')
    @patch('app.agent_logic.create_react_agent')
    @patch('app.agent_logic.RunnableWithMessageHistory')
    def test_online_mode_success(self, mock_runnable, mock_create_agent, 
                                  mock_load_tools, mock_ollama, mock_has_internet):
        """Test agent runs successfully in online mode"""
        mock_has_internet.return_value = True
        
        mock_agent_executor = MagicMock()
        mock_create_agent.return_value = mock_agent_executor
        
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"output": "Paris is the capital of France"}
        mock_runnable.return_value = mock_chain
        
        response, online = run_agent("What is the capital of France?", self.chat_history)
        
        assert response == "Paris is the capital of France"
        assert online is True
        mock_has_internet.assert_called_once()
        mock_load_tools.assert_called_once_with(["wikipedia", "ddg-search"])
    
    @patch('app.agent_logic.has_internet')
    @patch('app.agent_logic.ChatOllama')
    @patch('app.agent_logic.LLMChain')
    @patch('app.agent_logic.RunnableWithMessageHistory')
    def test_offline_mode_success(self, mock_runnable, mock_llm_chain, 
                                   mock_ollama, mock_has_internet):
        """Test agent runs successfully in offline mode"""
        mock_has_internet.return_value = False
        
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"text": "Based on my knowledge, Paris is the capital of France"}
        mock_runnable.return_value = mock_chain
        
        response, online = run_agent("What is the capital of France?", self.chat_history)
        
        assert response == "Based on my knowledge, Paris is the capital of France"
        assert online is False
        mock_has_internet.assert_called_once()
    
    @patch('app.agent_logic.has_internet')
    @patch('app.agent_logic.ChatOllama')
    @patch('app.agent_logic.load_tools')
    @patch('app.agent_logic.create_react_agent')
    @patch('app.agent_logic.RunnableWithMessageHistory')
    def test_chat_history_passed_correctly(self, mock_runnable, mock_create_agent,
                                            mock_load_tools, mock_ollama, mock_has_internet):
        """Test that chat history is properly passed to the chain"""
        mock_has_internet.return_value = True
        
        # Add some history
        self.chat_history.add_user_message("Hello")
        self.chat_history.add_ai_message("Hi there!")
        
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"output": "How can I help?"}
        mock_runnable.return_value = mock_chain
        
        run_agent("What's the weather?", self.chat_history)
        
        call_args = mock_chain.invoke.call_args[0][0]
        assert call_args["input"] == "What's the weather?"
        assert "User: Hello" in call_args["chat_history"]
        assert "Assistant: Hi there!" in call_args["chat_history"]

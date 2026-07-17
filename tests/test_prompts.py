import pytest
from app.prompts import react_prompt, offline_prompt

class TestPrompts:
    
    def test_react_prompt_has_required_variables(self):
        """Check that the ReAct prompt has all required variables"""
        try:
            formatted = react_prompt.format(
                chat_history="User: Hello\nAssistant: Hi there!",
                tools="Wikipedia, DuckDuckGo Search",
                tool_names="wikipedia, ddg-search",
                input="What is the capital of France?",
                agent_scratchpad="Thought: I need to search for this"
            )
            assert "You are a helpful assistant" in formatted
            assert "Wikipedia" in formatted
            assert "Final Answer" in formatted
        except KeyError as e:
            pytest.fail(f"Missing required variable: {e}")
    
    def test_react_prompt_missing_variable(self):
        """Test that the prompt fails gracefully when variables are missing"""
        with pytest.raises(KeyError):
            react_prompt.format(chat_history="")
    
    def test_offline_prompt_has_required_variables(self):
        """Check that the offline prompt has all required variables"""
        try:
            formatted = offline_prompt.format(
                chat_history="User: Hello\nAssistant: Hi!",
                input="Tell me about the weather"
            )
            assert "You are an offline answering assistant" in formatted
            assert "Gemma3:4b" in formatted
            assert "outdated" in formatted
        except KeyError as e:
            pytest.fail(f"Missing required variable: {e}")
    
    def test_prompt_contains_expected_keywords(self):
        """Verify critical content is in prompts"""
        assert "Question:" in react_prompt.template
        assert "Thought:" in react_prompt.template
        assert "Action:" in react_prompt.template
        assert "Observation:" in react_prompt.template
        assert "Final Answer:" in react_prompt.template
        assert "chat_history" in react_prompt.template
        
        assert "chat_history" in offline_prompt.template
        assert "Question:" in offline_prompt.template
        assert "outdated" in offline_prompt.template

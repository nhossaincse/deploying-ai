"""Gradio chat interface for the Atlas AI assistant."""

import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

from assignment_chat.main import get_graph
from assignment_chat.prompts import (
    contains_restricted_topic,
    is_prompt_injection,
    REFUSAL_MESSAGE,
    INJECTION_REFUSAL
)

# Load environment variables
load_dotenv('.env')
load_dotenv('.secrets')

# Initialize the agent graph
agent = get_graph()

# Memory management settings
MAX_HISTORY_MESSAGES = 20  # Threshold to trigger summarization
KEEP_RECENT_MESSAGES = 10  # Number of recent messages to keep after summarization


def summarize_old_messages(messages: list, keep_recent: int = KEEP_RECENT_MESSAGES) -> list:
    """Summarize older messages to preserve context while reducing tokens.
    
    When conversation exceeds MAX_HISTORY_MESSAGES, older messages are summarized
    into a single context message, keeping only the most recent messages intact.
    
    Args:
        messages: List of LangChain messages
        keep_recent: Number of recent messages to preserve
        
    Returns:
        List with summarized context + recent messages
    """
    if len(messages) <= keep_recent:
        return messages
    
    old_messages = messages[:-keep_recent]
    recent_messages = messages[-keep_recent:]
    
    # Create a summary request for old conversation
    conversation_text = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content[:200]}..."
        if len(m.content) > 200 else f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in old_messages
    ])
    
    summary_prompt = f"Summarize this conversation in 2-3 sentences, focusing on key topics discussed and any important context:\n\n{conversation_text}"
    
    try:
        # Use agent to generate summary
        summary_response = agent.invoke({"messages": [HumanMessage(content=summary_prompt)]})
        summary_text = summary_response['messages'][-1].content
        
        # Create a system-style context message
        context_message = AIMessage(content=f"[Previous conversation summary: {summary_text}]")
        
        return [context_message] + recent_messages
    except Exception:
        # If summarization fails, just truncate
        return recent_messages


def atlas_chat(message: str, history: list[dict]) -> str:
    """Process user message and return assistant response.
    
    Args:
        message: The user's input message
        history: List of previous messages in the conversation
        
    Returns:
        The assistant's response string
    """
    # Guardrail 1: Check for restricted topics
    if contains_restricted_topic(message):
        return REFUSAL_MESSAGE
    
    # Guardrail 2: Check for prompt injection attempts
    if is_prompt_injection(message):
        return INJECTION_REFUSAL
    
    # Convert Gradio history to LangChain messages
    langchain_messages = []
    
    for msg in history:
        if msg['role'] == 'user':
            langchain_messages.append(HumanMessage(content=msg['content']))
        elif msg['role'] == 'assistant':
            langchain_messages.append(AIMessage(content=msg['content']))
    
    # Memory management: summarize if conversation is too long
    if len(langchain_messages) > MAX_HISTORY_MESSAGES:
        langchain_messages = summarize_old_messages(langchain_messages)
    
    # Add current message
    langchain_messages.append(HumanMessage(content=message))
    
    # Prepare state for the agent
    state = {
        "messages": langchain_messages,
    }
    
    # Invoke the agent
    try:
        response = agent.invoke(state)
        # Get the last message from the response
        last_message = response['messages'][-1].content
        return last_message
    except Exception as e:
        return f"I encountered an error processing your request. Please try again."


# Create the Gradio chat interface
chat_interface = gr.ChatInterface(
    fn=atlas_chat,
    type="messages",
    title="Atlas - Your AI Assistant",
    description="Ask me about **weather**, **music reviews**, or **calculations**. I'm here to help!",
    examples=[
        "What's the weather like in Toronto?",
        "Find me some highly rated indie rock albums",
        "Calculate 25 * 4 + 100",
        "Convert 100 km to miles",
        "What albums are similar to Radiohead's style?",
    ],
    theme="soft"
)


if __name__ == "__main__":
    print("Starting Atlas Chat App...")
    chat_interface.launch()

from prompt_manager import PromptManager

def main():
    pm = PromptManager()
    
    # Mock data
    mock_chunks = [
        {"chunk_id": "123", "metadata": {"source": "test.md"}, "page_content": "To reset, click button."}
    ]
    
    rendered = pm.render_prompt(
        "support_v1.j2", 
        user_query="Help?", 
        context_chunks=mock_chunks
    )
    
    # Assertions
    assert "<context>" in rendered
    assert "To reset, click button." in rendered
    assert "test.md" in rendered
    assert "valid JSON" in rendered
    
    print("✅ Rendered prompt looks good.\n")
    print(rendered)

if __name__ == "__main__":
    main()
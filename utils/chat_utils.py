def process_chat(qa_chain, question, chat_history):
    try:
        response = qa_chain({
            "question": question,
            "chat_history": chat_history
        })
        return response
    except Exception as e:
        print(f"Error in process_chat: {e}")
        return None
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import threading

# Use FLAN-T5 Base for faster performance (approx 1GB download)
MODEL_NAME = "google/flan-t5-base"
tokenizer = None
model = None
model_lock = threading.Lock()

def init_t5():
    global tokenizer, model
    with model_lock:
        if model is None:
            print(f"Loading {MODEL_NAME} model...")
            tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
            # Force CPU for reliability if multiple models are loaded
            device = "cpu" 
            model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
            print(f"{MODEL_NAME} model loaded on {device}.")

def summarize_with_t5(text, min_words=300, max_words=500, custom_prompt=None):
    try:
        if model is None:
            init_t5()

        device = "cpu"

        # Detailed prompt for long summary
        if custom_prompt:
            input_text = custom_prompt.format(text=text)
        else:
            input_text = (
                "Write a detailed and well-structured summary of the following article. "
                "The summary should be between 300 and 500 words, covering all key points, "
                "important facts, and context in clear language:\n\n"
                f"{text}"
            )

        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        # Convert word counts to approximate token counts (1 word ~= 1.3 tokens)
        min_tokens = int(min_words * 1.3)
        max_tokens = int(max_words * 1.3)
        
        # Safety: If input is too short, don't force a long summary which causes hallucination
        input_length = inputs.input_ids.shape[1]
        if input_length < min_tokens:
            print(f"Warning: Input text is short ({input_length} tokens). Adjusting min_new_tokens to avoid hallucination.")
            min_tokens = min(50, input_length) # At least produce something, but don't force 400
            max_tokens = min(max_tokens, int(input_length * 1.5) + 100)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_tokens,
                min_new_tokens=min_tokens,
                num_beams=2,                 # Reduced from 4 to 2 for speed
                length_penalty=1.5,          # specific penalty
                no_repeat_ngram_size=3,
                early_stopping=True
            )
        
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary

    except Exception as e:
        print(f"T5 Summarization Error: {e}")
        return f"Error generating summary with T5: {str(e)}"


if __name__ == "__main__":
    test_text = "The Federal Reserve kept interest rates steady on Wednesday but took a major step toward lowering them in the coming months in a policy statement that gave a nod to inflation's decline. The central bank's latest move leaves its benchmark rate in a range between 5.25% and 5.5%, where it has been since July."
    print("Testing T5 Summarization...")
    print(summarize_with_t5(test_text))

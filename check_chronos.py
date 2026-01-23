
try:
    import torch
    from transformers import AutoModelForSeq2SeqLM
    
    print("Transformers imported successfully.")
    
    model_name = "amazon/chronos-t5-tiny"
    print(f"Loading {model_name}...")
    # Just load config/model to enable check without downloading huge weights if cached
    # unlikely to fail if transformers is there
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
    print("Model loaded successfully.")
    
except ImportError:
    print("Transformers not installed.")
except Exception as e:
    print(f"Error loading model: {e}")

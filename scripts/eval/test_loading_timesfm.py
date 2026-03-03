import os
import torch
import numpy as np
from transformers import TimesFmModelForPrediction, TimesFmConfig

def test_loading():
    model_id = "google/timesfm-2.5-200m-transformers"
    print(f"Testing direct loading of {model_id} using TimesFmModelForPrediction...")
    try:
        # Try to load the config and force model_type if necessary
        # config = TimesFmConfig.from_pretrained(model_id)
        # print(f"Config loaded: {config.model_type}")
        
        model = TimesFmModelForPrediction.from_pretrained(model_id, trust_remote_code=True)
        print("Model loaded successfully using TimesFmModelForPrediction!")
        
        # Dummy input: (batch, seq_len)
        # Context length for TimesFM 2.5 is usually 512 or up to 2048
        # We also need to provide 'patch_view' etc. depending on the implementation
        # Let's check the signature of forward
        import inspect
        sig = inspect.signature(model.forward)
        print(f"Forward signature: {sig}")
        
    except Exception as e:
        print(f"Error during loading: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_loading()

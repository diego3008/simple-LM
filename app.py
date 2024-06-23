from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Try to load the chatbot model
try:
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    app.logger.info("Chatbot model loaded successfully")
except Exception as e:
    app.logger.error(f"Error loading chatbot model: {str(e)}")
    tokenizer = None
    model = None

@app.route('/chat', methods=['POST'])
def chat():
    if model is None or tokenizer is None:
        return jsonify({'error': 'Chatbot model not available'}), 503

    try:
        user_message = request.json['message']
        app.logger.info(f"Received message: {user_message}")

        # Encode the input and add end of string token
        input_ids = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors='pt')

        # Generate a response
        chat_history_ids = model.generate(
            input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            # no_repeat_ngram_size=3,
            do_sample=True,
            top_k=100,
            top_p=0.7,
            temperature=0.8
        )

        # Decode the response
        response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        
        return jsonify({'response': response})
    except KeyError:
        app.logger.error("KeyError: 'message' not found in request JSON")
        return jsonify({'error': 'Invalid request. "message" key is required.'}), 400
    except Exception as e:
        app.logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    health_status = {
        'status': 'healthy',
        'chatbot_available': model is not None and tokenizer is not None
    }
    return jsonify(health_status)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

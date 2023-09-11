import torch
import torchtext
import spacy
from flask import Flask, request, jsonify, render_template
import pickle
import requests

# ina ro por kon ke jelo bot ha ro begiri, kilooii hamle nakonan be web pagetoon!
# https://www.google.com/recaptcha/admin
RECAPTCHA_SITE_KEY = '6LeieQUoAAAAAPbRpUV764zt70q0Jg0echYVucuj'
RECAPTCHA_SECRET_KEY = '6LeieQUoAAAAAL17x0JsV0oMP_Ip7ZSNrmdYHDCC'

class RNN(torch.nn.Module):
    '''
    Basic RNN (LSTM) with 1 hidden layer.
    '''
    
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(input_dim, embedding_dim)
        self.rnn = torch.nn.LSTM(embedding_dim,
                                 hidden_dim)        
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        
    def forward(self, answer):
        embedded = self.embedding(answer)
        output, (hidden, cell) = self.rnn(embedded)
        hidden.squeeze_(0)
        output = self.fc(hidden)
        return output

def predict_BotOrNot(model, text):
    '''
    Function to test a text-classifying model on any given text
    '''
    model.eval()
    
    tokenized = [tok.text for tok in nlp.tokenizer(text)]
    indexed = [answer_field.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(DEVICE)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    prediction = torch.sigmoid(model(tensor))
    
    return (prediction[0][0].item())

app = Flask(__name__)

with open("answer_field.pkl", "rb") as f:
    answer_field = pickle.load(f)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
params = {
    'emb_dim': 256,
    'hid_dim': 256,
    'out_dim': 1,
    'n_layers': 2,
    'dropout': 0.35,
    'lr': 0.01,
    'epochs': 13,
    'batch_size': 128,
    'test_prc': 0.2,
}
input_dim = len(answer_field.vocab)

model = RNN(input_dim=input_dim,
            embedding_dim=params['emb_dim'],
            hidden_dim=params['hid_dim'],
            output_dim=params['out_dim']
)
model.load_state_dict(torch.load('model.pth', map_location=DEVICE))
model.to(DEVICE)

nlp = spacy.blank("en")

@app.route("/", methods=["GET", "POST"])
def index():
    result_text = None
    recaptcha_enabled = len(RECAPTCHA_SECRET_KEY) > 0 and len(RECAPTCHA_SITE_KEY) > 0

    if request.method == "POST":
        if recaptcha_enabled:
            recaptcha_response = request.form['g-recaptcha-response']
            payload = {
                'secret': RECAPTCHA_SECRET_KEY,
                'response': recaptcha_response
            }
            result = requests.post('https://www.google.com/recaptcha/api/siteverify', data=payload).json()
        else:
            result = { 'success': True }

        if result['success']:
            text_to_check = request.form["text_to_check"]
            result_text = predict_BotOrNot(model, text_to_check)
        else:
            result_text = "reCAPTCHA verification failed."

    return render_template("index.html", 
                           result_text = result_text, 
                           recaptcha_enabled = recaptcha_enabled, 
                           site_key = RECAPTCHA_SITE_KEY)

'''
@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.json['text']
        prediction = predict_BotOrNot(model, text)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
'''

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
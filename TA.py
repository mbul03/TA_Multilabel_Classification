import streamlit as st
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

# Cache the model and tokenizer to avoid reloading
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("model/indobert_tokenizer")
    model = AutoModel.from_pretrained("model/indobert_model")
    return tokenizer, model

tokenizer, bert_model = load_model_and_tokenizer()

# Define the function to get sentence embedding
def get_sentence_embedding(text: str, max_sequence_length: int):
    assert tokenizer is not None, "tokenizer not initialized"
    assert bert_model is not None, "bert_model not initialized"

    text_input = f"[CLS] {text.lower()} [SEP] [MASK]"
    tokenized_text = tokenizer.tokenize(text_input)
    segments = [1] * len(tokenized_text)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments])

    with torch.no_grad():
        encoded_layers = bert_model(tokens_tensor, segments_tensors)

    last_hidden_state = encoded_layers.last_hidden_state
    token_embeddings = torch.stack([last_hidden_state])
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1, 0, 2)

    token_vecs = last_hidden_state[0]
    sentence_embedding = [tensor for tensor in token_vecs]
    mask_token = sentence_embedding.pop()

    if len(sentence_embedding) > max_sequence_length:
        last_token = sentence_embedding.pop()
        sentence_embedding = sentence_embedding[:max_sequence_length-1]
        sentence_embedding.append(last_token)
    
    sentence_embedding = [mask_token]*(max_sequence_length-len(sentence_embedding)) + sentence_embedding

    return np.array(sentence_embedding)

# Define Attention Mechanism
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output):
        attention_weights = torch.tanh(self.attention(lstm_output)).squeeze(2)
        attention_weights = torch.softmax(attention_weights, dim=1)
        weighted_output = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)
        return weighted_output, attention_weights

# Define BiLSTM with Attention
class BiLSTM_Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_size * 2)
        self.dropout = nn.Dropout(p=0.3)  # Dropout layer
        self.fc = nn.Linear(hidden_size * 2, 256)  # Reduced size
        self.bn = nn.BatchNorm1d(256)  # Batch Normalization
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm1(x, (h0, c0))
        attn_output, _ = self.attention(out)

        out = self.dropout(attn_output)
        out = self.fc(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        return out

# Instantiate the model
input_size = 768
hidden_size = 256
num_layers = 1
num_classes = 39

@st.cache_resource
def load_news_model():
    model = BiLSTM_Attention(input_size, hidden_size, num_layers, num_classes)
    model.load_state_dict(torch.load("model/Hybrid_Hyper1.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

news_model = load_news_model()

# Define the Streamlit app
st.title("Multilabel Klasifikasi Teks Berita dengan Bi-LSTM, Attention, dan BERT")

st.subheader("Input Teks")
input_text = st.text_area("Masukkan teks berita di sini")

if st.button("Klasifikasi"):
    if input_text:
        predict_embed = torch.tensor(np.array([get_sentence_embedding(input_text, 32)]), dtype=torch.float32)
        with torch.no_grad():
            predict_result = news_model(predict_embed)
        predict_class_result = np.round(predict_result).numpy()

        output_labels = ['olahraga',
 'ekonomi',
 'kecelakaan',
 'kriminalitas',
 'bencana',
 'bulutangkis',
 'voli',
 'basket',
 'tenis',
 'pembunuhan',
 'pencurian',
 'gempa',
 'kebakaran',
 'tsunami',
 'gunung_meletus',
 'banjir',
 'puting_beliung',
 'kekeringan',
 'abrasi',
 'longsor',
 'pendidikan',
 'teknologi',
 'politik',
 'kesehatan',
 'sains',
 'bisnis',
 'bisnis kecil',
 'media',
 'pasar',
 'seni',
 'desain',
 'musik',
 'tari',
 'film',
 'teater',
 'golf',
 'sepakbola',
 'baseball',
 'hoki']
        labels = [output_labels[i] for i in range(len(output_labels)) if predict_class_result[0][i] == 1]
        st.subheader("Output label hasil klasifikasi")
        st.write(", ".join(labels))
    else:
        st.write("Masukkan teks terlebih dahulu.")

st.markdown("---")
st.markdown("**Dibuat oleh Marsyavero Charisyah Putra**")

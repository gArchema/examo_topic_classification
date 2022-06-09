import os

import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

text_labels = ['collections', 'apis', 'classes_methods', 'tools', 'library', 'patterns', 'framework', 'databases']


class QuestionClassifierModel(torch.nn.Module):

    def __init__(self, Pretrained_model, cpu_gpu):
        super().__init__()
        self.device = torch.device(cpu_gpu)
        self.Pretrained_model = Pretrained_model.to(self.device)

        for param in self.Pretrained_model.parameters():
            param.requires_grad = False

        self.linear_0 = torch.nn.Linear(384, 192, device=self.device)

        self.last_hidden_layer = torch.nn.Linear(192, 8, device=self.device)

        self.activation_0 = torch.nn.ReLU()

        self.activation_last = torch.nn.Sigmoid()

        self.layer_norm = torch.nn.LayerNorm(192, device=self.device)

        self.dropout = torch.nn.Dropout(p=0.3)

    def forward(self, sentence):
        with torch.no_grad():
            out = self.Pretrained_model.encode(sentence, convert_to_tensor=True)

        out = self.linear_0(out)
        out = self.activation_0(out)

        out = self.dropout(out)
        out = self.layer_norm(out)

        out = self.last_hidden_layer(out)

        return self.activation_last(out)


embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

the_model = QuestionClassifierModel(embedding_model, device)
the_model.load_state_dict(torch.load(os.getenv('MODEL_PATH'), map_location=device))


def get_index(test_question):
    return text_labels[the_model(test_question).argmax().item()]

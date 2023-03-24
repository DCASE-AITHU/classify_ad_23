from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2
from speechbrain.lobes.models.ECAPA_TDNN import Classifier, BatchNorm1d, AttentiveStatisticsPooling
from speechbrain.nnet.losses import LogSoftmaxWrapper, AdditiveAngularMargin
from speechbrain.processing.features import InputNormalization
import torch

class W2V(torch.nn.Module):
    def __init__(self, embedding_dim=256, output_dim = 119):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.w2v_model = HuggingFaceWav2Vec2("facebook/wav2vec2-xls-r-300m", "./cache/", False, freeze=False)
        self.pooling_module = AttentiveStatisticsPooling(self.w2v_model.output_size, attention_channels=self.embedding_dim//2)
        self.pooling_module_bn = BatchNorm1d(input_size=self.w2v_model.output_size * 2)
        self.fc = torch.nn.Linear(self.w2v_model.output_size * 2, self.embedding_dim)
        
        self.linear = Classifier(input_size=self.embedding_dim, out_neurons=self.output_dim)
        ### loss definition
        self.loss = LogSoftmaxWrapper(AdditiveAngularMargin(margin=0.2, scale=30))

    def embedding(self, waveform):
        output = self.w2v_model(waveform)
        output = self.fc( self.pooling_module_bn( self.pooling_module(output.transpose(1,2)) ).transpose(1,2) )
        return output.squeeze(1)
    
    def forward(self, waveform, labels=None):
        output = self.embedding(waveform)
        if self.training:
            output = self.linear(output)
            output = self.loss(output, labels.to(torch.long))
        else:
            output = {'embedding': output}
        return output

        
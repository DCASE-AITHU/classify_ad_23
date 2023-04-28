from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2
from speechbrain.lobes.models.ECAPA_TDNN import Classifier, BatchNorm1d, AttentiveStatisticsPooling
from speechbrain.nnet.losses import LogSoftmaxWrapper, AdditiveAngularMargin
from speechbrain.processing.features import InputNormalization
import torch
from transformers import AutoProcessor, HubertModel

class W2V(torch.nn.Module):
    def __init__(self, model_name="wav2vec_300m", embedding_dim=256, output_dim = 119, weighted_sum=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.model_name = model_name
        if model_name == "wav2vec_300m":
            self.w2v_model = HuggingFaceWav2Vec2("facebook/wav2vec2-xls-r-300m", "./cache/", False, freeze=False, weighted_sum=weighted_sum)
        elif model_name == "wav2vec_1b":
            self.w2v_model = HuggingFaceWav2Vec2("facebook/wav2vec2-xls-r-1b", "./cache/", False, freeze=False, weighted_sum=weighted_sum)
        elif model_name == "wav2vec_2b":
            self.w2v_model = HuggingFaceWav2Vec2("facebook/wav2vec2-xls-r-2b", "./cache/", False, freeze=False, weighted_sum=weighted_sum)
        elif model_name == "hubert_base":
            self.w2v_model = HuggingFaceWav2Vec2("facebook/hubert-base-ls960", "./cache/", False, freeze=False, weighted_sum=weighted_sum)
        elif model_name == "hubert_large":
            self.w2v_model = HuggingFaceWav2Vec2("facebook/hubert-large-ll60k", "./cache/", False, freeze=False, weighted_sum=weighted_sum)
        elif model_name == "hubert_xlarge":
            self.w2v_model = HuggingFaceWav2Vec2("facebook/hubert-xlarge-ll60k", "./cache/", False, freeze=False, weighted_sum=weighted_sum)
        elif model_name == "unispeech_base":
            self.w2v_model = HuggingFaceWav2Vec2("microsoft/unispeech-sat-base", "./cache/", False, freeze=False, weighted_sum=weighted_sum)
        elif model_name == "unispeech_large":
            self.w2v_model = HuggingFaceWav2Vec2("microsoft/unispeech-sat-large", "./cache/", False, freeze=False, weighted_sum=weighted_sum)
        elif model_name == "wavlm_base":
            self.w2v_model = HuggingFaceWav2Vec2("microsoft/wavlm-base-plus", "./cache/", False, freeze=False, weighted_sum=weighted_sum)
        elif model_name == "wavlm_large":
            self.w2v_model = HuggingFaceWav2Vec2("microsoft/wavlm-large", "./cache/", False, freeze=False, weighted_sum=weighted_sum)
        self.pooling_module = AttentiveStatisticsPooling(self.w2v_model.output_size, attention_channels=self.embedding_dim//2)
        self.pooling_module_bn = BatchNorm1d(input_size=self.w2v_model.output_size * 2)
        self.fc = torch.nn.Linear(self.w2v_model.output_size * 2, self.embedding_dim)
        
        self.linear = Classifier(input_size=self.embedding_dim, out_neurons=self.output_dim)
        ### loss definition
        self.loss = LogSoftmaxWrapper(AdditiveAngularMargin(margin=0.2, scale=30))

    def embedding(self, waveform):
#        if self.model_name == "wav2vec":
#            output = self.w2v_model(waveform)
#        elif "hubert" in self.model_name:
#            input_values = processor(waveform, return_tensors="pt").input_values
#            output = model(input_values)
#            output = output.last_hidden_state
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

        

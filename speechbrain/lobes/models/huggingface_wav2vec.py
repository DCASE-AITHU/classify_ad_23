"""This lobe enables the integration of huggingface pretrained wav2vec2 models.

Reference: https://arxiv.org/abs/2006.11477
Reference: https://arxiv.org/abs/1904.05862
Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Authors
 * Titouan Parcollet 2021
 * Boumadane Abdelmoumene 2021
"""

import torch
import torch.nn.functional as F
from torch import nn

# We check if transformers is installed.
try:
    from transformers import Wav2Vec2Model, HubertModel, UniSpeechSatModel, WavLMModel, WhisperModel
    from transformers import Wav2Vec2Config, HubertConfig, UniSpeechSatConfig, WavLMConfig, WhisperConfig
    from transformers import Wav2Vec2FeatureExtractor
    from transformers import AutoModelForAudioClassification
except ImportError:
    print(
        "Please install transformer from HuggingFace to use wav2vec2/Hubert !"
    )


HF_models = {"wav2vec2": Wav2Vec2Model, "hubert": HubertModel, "unispeech": UniSpeechSatModel, "wavlm": WavLMModel, "whisper": WhisperModel}
HF_config = {"wav2vec2": Wav2Vec2Config, "hubert": HubertConfig, "unispeech": UniSpeechSatConfig, "wavlm": WavLMConfig, "whisper": WhisperConfig}


class HuggingFaceWav2Vec2(nn.Module):
    """This lobe enables the integration of HuggingFace
    pretrained wav2vec2.0/Hubert models.

    Source paper wav2vec2.0: https://arxiv.org/abs/2006.11477
    Source paper Hubert: https://arxiv.org/abs/2106.07447
    Transformer from HuggingFace needs to be installed:
    https://huggingface.co/transformers/installation.html

    The model can be used as a fixed feature extractor or can be finetuned. It
    will download automatically the model from HuggingFace.

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "facebook/wav2vec2-large-lv60"
    save_path : str
        Path (dir) of the downloaded model.
    output_norm : bool (default: True)
        If True, a layer_norm (affine) will be applied to the output obtained
        from the wav2vec model.
    freeze : bool (default: True)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    freeze_feature_extractor :  bool (default: False)
        When freeze = False and freeze_feature_extractor True, the featue_extractor module of the model is Frozen. If False
        all the wav2vec model will be trained including featue_extractor module.
    pretrain : bool (default: True)
        If True, the model is pretrained with the specified source.
        If False, the randomly-initialized model is instantiated.
    apply_spec_augment : bool (default: False)
        If True, the model will apply spec augment on the output of feature extractor
        (inside huggingface Wav2VecModel() class).
        If False, the model will not apply spec augment. We set this to false to prevent from doing it twice.
    Example
    -------
    >>> inputs = torch.rand([10, 600])
    >>> model_hub = "facebook/wav2vec2-base-960h"
    >>> save_path = "savedir"
    >>> model = HuggingFaceWav2Vec2(model_hub, save_path)
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([10, 1,  768])
    """

    def __init__(
        self,
        source,
        save_path,
        output_norm=True,
        freeze=True,  # 是否固定整个模型参数
        freeze_feature_extractor=False,  # 是否固定特征提取器的参数
        pretrain=True,
        apply_spec_augment=False,
        hidden_concate=False,
        weighted_sum=True,  # 是否使用weighted_sum
    ):
        super().__init__()

        # Download the extractor from HuggingFace.
        # The extractor is only used to retrieve the normalisation
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            source, cache_dir=save_path
        )

        # Select specific self-supervised loader (eg. Wav2Vec2, Hubert)
        if "hubert" in source:
            config = HF_config.get("hubert")
            model = HF_models.get("hubert")
        elif "wav2vec" in source:
            config = HF_config.get("wav2vec2")
            model = HF_models.get("wav2vec2")
        elif "unispeech" in source:
            config = HF_config.get("unispeech")
            model = HF_models.get("unispeech")
        elif "wavlm" in source:
            config = HF_config.get("wavlm")
            model = HF_models.get("wavlm")
        elif "whisper" in source:
            config = HF_config.get("whisper")
            model = HF_models.get("whisper")
        # Download the model from HuggingFace.
        # if pretrain is False, we do not download the pretrained weights
        # it it is True, we download and load them.
        if not (pretrain):
            config = config.from_pretrained(source, cache_dir=save_path)
            config = config.from_pretrained(source, cache_dir=save_path)
            self.model = model(config)
        else:
            self.model = model.from_pretrained(source, cache_dir=save_path)
        # self.model = model.from_pretrained(source, cache_dir=save_path)
        # self.model = model.from_pretrained(source)

        # set apply_spec_augment
        self.model.config.apply_spec_augment = apply_spec_augment

        # We check if inputs need to be normalized w.r.t pretrained wav2vec2
        self.normalize_wav = self.feature_extractor.do_normalize

        self.freeze = freeze
        self.freeze_feature_extractor = freeze_feature_extractor
        self.output_norm = output_norm
        if self.freeze:
            self.model.eval()
        else:
            self.model.train()
            if self.freeze_feature_extractor:
                self.model.feature_extractor._freeze_parameters()
        self.hidden_concate = hidden_concate
        self.output_size = self.model.config.hidden_size
        self.weighted_sum = weighted_sum
        if self.weighted_sum:
            # 不同层输出乘上可学习的参数，叠加
            self.feature_weight = nn.Parameter(torch.zeros(1 + self.model.config.num_hidden_layers))
        # if self.hidden_concate:
        #    self.output_size = self.output_size * (1+self.model.config.num_hidden_layers)

    def forward(self, wav):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """

        # If we freeze, we simply remove all grads and features from the graph.
        if self.freeze:
            with torch.no_grad():
                return self.extract_features(wav).detach()

        return self.extract_features(wav)

    def extract_features(self, wav):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """

        if self.normalize_wav:
            wav = F.layer_norm(wav, wav.shape)

        # Extract wav2vec output
        if self.weighted_sum:
            output = self.model(wav, output_hidden_states=True)
            bs, length, dim = output['hidden_states'][0].shape
            out = torch.cat(output['hidden_states'], dim=-1)
            out = out.reshape(bs, length, dim, -1)
            out = torch.transpose(out, 0, 3)
            norm_weights = F.softmax(self.feature_weight, dim=-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            out = norm_weights * out  # 不同层的输出加权，每个样本乘的系数相同
            out = torch.sum(out, 0)
            out = out.permute(2, 0, 1)
        else:
            out = self.model(wav)[0]

        # We normalize the output if required
        if self.output_norm:
            out = F.layer_norm(out, out.shape)

        return out

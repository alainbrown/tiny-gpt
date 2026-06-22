from transformers import PretrainedConfig, PreTrainedModel
from model import Model

class TinyGPTConfig(PretrainedConfig):
    model_type = "tiny_gpt"
    
    def __init__(self, context_size=32, vocab_size=1024, d_model=64, n_layers=4, n_heads=4, dropout=0.1, **kwargs):
        self.context_size = context_size
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        super().__init__(**kwargs)

class TinyGPTForCausalLM(PreTrainedModel):
    config_class = TinyGPTConfig

    def __init__(self, config):
        super().__init__(config)
        self.core_model = Model(
            context_size=config.context_size,
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            dropout=config.dropout
        )

    def forward(self, x):
        return self.core_model(x)

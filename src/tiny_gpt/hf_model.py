from transformers import PretrainedConfig, PreTrainedModel
from tiny_gpt.model import Model

class TinyGPTConfig(PretrainedConfig):
    model_type = "tiny_gpt"
    
    def __init__(self, context_size=32, vocab_size=1024, d_model=64, n_layers=4, n_heads=4, dropout=0.1, tie_word_embeddings=True, **kwargs):
        self.context_size = context_size
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

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
        
        # Proper Hugging Face initialization (v5.0+)
        self.post_init()

    def get_input_embeddings(self):
        return self.core_model.token_embedding

    def set_input_embeddings(self, value):
        self.core_model.token_embedding = value

    def get_output_embeddings(self):
        return self.core_model.linear

    def set_output_embeddings(self, new_embeddings):
        self.core_model.linear = new_embeddings

    def forward(self, x):
        return self.core_model(x)

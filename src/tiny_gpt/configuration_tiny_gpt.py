from transformers import PretrainedConfig


class TinyGPTConfig(PretrainedConfig):
    model_type = "tiny_gpt"

    def __init__(
        self,
        context_size=32,
        vocab_size=1024,
        d_model=64,
        n_layers=4,
        n_heads=4,
        dropout=0.1,
        tie_word_embeddings=True,
        use_cache=False,
        **kwargs,
    ):
        self.context_size = context_size
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.hidden_size = d_model
        self.num_hidden_layers = n_layers
        self.num_attention_heads = n_heads
        self.max_position_embeddings = context_size
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            use_cache=use_cache,
            **kwargs,
        )

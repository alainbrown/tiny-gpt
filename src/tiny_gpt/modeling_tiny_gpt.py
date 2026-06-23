import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput

from .configuration_tiny_gpt import TinyGPTConfig
from .model import GPTModel


class TinyGPTForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = TinyGPTConfig
    main_input_name = "input_ids"
    _tied_weights_keys = {"core_model.linear.weight": "core_model.token_embedding.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.core_model = GPTModel(
            context_size=config.context_size,
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.core_model.token_embedding

    def set_input_embeddings(self, value):
        self.core_model.token_embedding = value

    def get_output_embeddings(self):
        return self.core_model.linear

    def set_output_embeddings(self, new_embeddings):
        self.core_model.linear = new_embeddings

    def forward(self, input_ids=None, labels=None, **kwargs):
        if input_ids is None:
            raise ValueError("input_ids must be provided")

        logits = self.core_model(input_ids)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return CausalLMOutput(loss=loss, logits=logits)

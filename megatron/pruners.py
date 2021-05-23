import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import megatron
from megatron.mpu.layers import RowParallelLinear, ColumnParallelLinear

sparsity = 10

def transformers_pruning(layers_to_prune, model, LevelPruningMethod, prune_bias):
    for layers in model.module.module.bert.encoder.layer:
        if 'attention' in layers_to_prune:
            for layer in layers.attention.children():
                for l in layer.children():
                    if type(l) == nn.Linear:
                        LevelPruningMethod.apply(l, 'weight')
                        if (prune_bias):
                            LevelPruningMethod.apply(l, 'bias')
        if 'intermediate' in layers_to_prune:
            for l in layers.intermediate.children():
                if type(l) == nn.Linear:
                    LevelPruningMethod.apply(l, 'weight')
                    if (prune_bias):
                        LevelPruningMethod.apply(l, 'bias')
        if 'output' in layers_to_prune:
            for l in layers.output.children():
                if type(l) == nn.Linear:
                    LevelPruningMethod.apply(l, 'weight')
                    if (prune_bias):
                        LevelPruningMethod.apply(l, 'bias')
    return model

def transformers_calculate_sparsity(model):
    zero = 0
    total = 0
    for x in model.module.module.bert.encoder.layer:
        for y in x.children():
            for l in y.children():
                if 'Attention' in str(type(l)) or 'Output' in str(type(l)):
                    for a in l.children():
                        if type(a) == nn.Linear:
                            zero += (a.weight == 0).sum()
                            total += a.weight.view(-1).size()[0]
                elif type(l) == nn.Linear:
                    zero += (l.weight == 0).sum()
                    total += l.weight.view(-1).size()[0]
    return zero, total

def megatron_pruning(layers_to_prune, model, LevelPruningMethod, prune_bias):
    for layers in model.module.module.language_model.transformer.layers:
        if 'attention' in layers_to_prune:
            for l in layers.attention.children():
                if type(l) == ColumnParallelLinear or type(l) == RowParallelLinear:
                    LevelPruningMethod.apply(l, 'weight')
                    if (prune_bias):
                        LevelPruningMethod.apply(l, 'bias')
        if 'mlp' in layers_to_prune:
            for l in layers.mlp.children():
                if type(l) == ColumnParallelLinear or type(l) == RowParallelLinear:
                    LevelPruningMethod.apply(l, 'weight')
                    if (prune_bias):
                        LevelPruningMethod.apply(l, 'bias')
    return model

def megatron_calculate_sparsity(model):
    zero = 0
    total = 0
    # 0 1 2 ... 23
    for x in model.module.module.language_model.transformer.layers:
        # input_layernorm, attention, post_attention_layernorm, mlp
        for y in x.children():
            if 'Attention' in str(type(y)) or 'Output' in str(type(y)):
                # layers within attention/mlp
                for l in y.children():
                    if type(l) == ColumnParallelLinear or type(l) == RowParallelLinear:
                        zero += (l.weight == 0).sum()
                        total += l.weight.view(-1).size()[0]
    return zero, total

class LevelPruning():
    """ Level pruning """

    def __init__(self, model, sparsity=0.25, prune_bias=True, model_provider="transformers"):
        self.sparsity = sparsity
        self.step_sparsity = sparsity
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.device_count() > 1 and model_provider not in "megatron":
            model = nn.DataParallel(model)

        self.model = model.to(self.device)
        self.prune_bias = prune_bias
        self.level = 10
        self.model_provider = model_provider

    @staticmethod
    def calculate_cutoff(sparsity: float, flat_params):
        """ Compute cutoff value based on sparsity """
        assert flat_params.dim() == 1

        with torch.no_grad():
            cutoff_index = round(sparsity * flat_params.size()[0]) - 1
            values, _ = torch.sort(torch.abs(flat_params))
            cutoff = values[cutoff_index]

        return cutoff

    def clone_params(self):
        """ Copy all tracked params, such that they we can rewind to them later """
        self.cloned = self.model.state_dict()

    def rewind(self):
        for i in self.model.named_modules():
            if (i[0]+'.weight_orig') in self.cloned:
                i[-1].weight = self.cloned[(i[0]+'.weight_orig')]

    def step(self, layers_to_prune=['attention']):
        """ Update the pruning masks """
        # iterate through 24 layers
        if not self.level:
            print('Hit highest possible sparsity.')
            return

        if self.model_provider in "transformers":
            self.model = transformers_pruning(layers_to_prune, self.model, self.LevelPruningMethod, self.prune_bias)
        elif self.model_provider in "megatron":
            self.model = megatron_pruning(layers_to_prune, self.model, self.LevelPruningMethod, self.prune_bias)

        self.step_sparsity += self.sparsity
        self.level = self.level - 1 if self.level > 0 else 0

        global sparsity
        sparsity = self.level

    def calculate_sparsity(self):
        """ Calculate global sparsity """
        zero = None
        total = None
        if self.model_provider in "transformers":
            zero, total = transformers_calculate_sparsity(self.model)
        elif self.model_provider in "megatron":
            zero, total = megatron_calculate_sparsity(self.model)
        return zero.item()/total

    class LevelPruningMethod(prune.BasePruningMethod):
        PRUNING_TYPE = 'unstructured'

        def compute_mask(self, t, default_mask):
            global sparsity
            flat_params = t.view(-1)
            # Steps by 10% for each pruning iteration
            cutoff = LevelPruning.calculate_cutoff(1/sparsity, flat_params)
            mask = torch.where(torch.abs(t) < cutoff,
                               torch.zeros_like(t), default_mask)
            return torch.nn.Parameter(mask)

from cybertron.BaseModel import BaseModel
from cybertron.GRUC import GRUC
from cybertron.LeapGRU import LeapGRU
from cybertron.Transformer import Transformer
from cybertron.LSTMC import LSTMC
from cybertron.RLModel import RLModel
from cybertron.RLModelKL import RLModelKL
from cybertron.SkipLSTM import SkipLSTM


ALL_MODELS = {
    cls.__name__: cls for cls in BaseModel.__subclasses__()
}
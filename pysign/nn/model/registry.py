from ...utils import RegistryBase
import typing as _typing
from torch.nn import Module


class EncoderRegistry(RegistryBase):
    @classmethod
    def register_encoder(cls, encoder_name: str) -> _typing.Callable[
        [_typing.Type[Module]], _typing.Type[Module]
    ]:
        def register_encoder_cls(encoder: _typing.Type[Module]):
            if not issubclass(encoder, Module):
                raise TypeError
            else:
                cls[encoder_name] = encoder
                return encoder

        return register_encoder_cls

    @classmethod
    def get_encoder(cls, encoder_name: str) -> _typing.Type[Module]:
        if encoder_name not in cls:
            raise NotImplementedError('Unknown encoder', encoder_name)
        else:
            return cls[encoder_name]
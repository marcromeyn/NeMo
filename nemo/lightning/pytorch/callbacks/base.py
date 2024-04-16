import inspect
from collections import defaultdict

import pytorch_lightning as L


class CallbackConnector:
    def __init__(self, callbacks) -> None:
        self.callbacks = defaultdict(list)
        self.add(*callbacks)

    def add(self, *callbacks) -> "CallbackConnector":
        megatron_methods = {
            m for m in dir(MegatronCallback) if m.startswith("on") and not hasattr(L.Callback, m)
        }

        for callback in callbacks:
            if isinstance(callback, CallbackConnector):
                # Handle CallbackConnector instance: merge its callbacks
                for event_name, event_callbacks in callback.callbacks.items():
                    self.callbacks[event_name].extend(event_callbacks)
            else:
                for method in megatron_methods:
                    if hasattr(callback, method) and callable(getattr(callback, method)):
                        self.callbacks[method].append(callback)

        return self

    def event(self, name: str, *args, **kwargs) -> None:
        for callback in self.callbacks.get(name, []):
            callback_method = getattr(callback, name, None)
            if callable(callback_method):
                # Inspect the callback method to determine accepted arguments
                sig = inspect.signature(callback_method)

                # Filter args based on the callback method's parameters
                filtered_args = [
                    arg
                    for arg, param in zip(args, sig.parameters.values())
                    if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
                ]

                # Filter kwargs based on the callback method's parameters
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

                # Call the method with only the accepted arguments
                callback_method(*filtered_args, **filtered_kwargs)


class MegatronCallback(L.Callback):
    def on_megatron_step_start(self, *args, **kwargs) -> None:
        ...

    def on_megatron_microbatches_start(self, *args, **kwargs) -> None:
        ...

    def on_megatron_microbatches_callback(self, *args, **kwargs) -> None:
        ...

    def on_megatron_microbatches_end(self, *args, **kwargs) -> None:
        ...

    def on_megatron_reduce_microbatches_start(self, *args, **kwargs) -> None:
        ...

    def on_megatron_reduce_microbatches_end(self, *args, **kwargs) -> None:
        ...

    def on_megatron_log_step_end(self, *args, **kwargs) -> None:
        ...

    def on_megatron_step_end(self, *args, **kwargs) -> None:
        ...


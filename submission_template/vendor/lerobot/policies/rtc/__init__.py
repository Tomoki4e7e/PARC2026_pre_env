# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real-Time Chunking (RTC) utilities for action-chunking policies."""

from .action_interpolator import ActionInterpolator
from .action_queue import ActionQueue
from .configuration_rtc import RTCConfig
from .latency_tracker import LatencyTracker
from .relative import reanchor_relative_rtc_prefix


_LAZY_IMPORTS = {
    "RTCProcessor": "modeling_rtc",
}

def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        from importlib import import_module

        module = import_module(f".{_LAZY_IMPORTS[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
__all__ = [
    "ActionInterpolator",
    "ActionQueue",
    "LatencyTracker",
    "RTCConfig",
    "RTCProcessor",
    "reanchor_relative_rtc_prefix",
]

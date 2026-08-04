# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from .configuration_tdmpc import TDMPCConfig


_LAZY_IMPORTS = {
    "TDMPCPolicy": "modeling_tdmpc",
    "make_tdmpc_pre_post_processors": "processor_tdmpc",
}

def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        from importlib import import_module

        module = import_module(f".{_LAZY_IMPORTS[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
__all__ = ["TDMPCConfig", "TDMPCPolicy", "make_tdmpc_pre_post_processors"]

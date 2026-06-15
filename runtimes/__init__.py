"""Runtimes — concrete numerical backends for component specs.

Each subpackage is one port (torch_port, openvino_port, onnx_port, ...).
The torch_port is the ground-truth reference; the others must match it.
"""

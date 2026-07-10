# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinMLSession - ONNX Runtime session manager with WinML EP integration."""

from ..ep_path import DirectorySource, EPEntry
from .ep_device import (
    _SHORT_TO_FULL,
    DEVICE_TO_DEVICE_TYPE,
    DEVICE_TYPE_TO_DEVICE,
    EP_DEVICE_SPECS,
    VALID_DEVICES,
    VALID_EPS,
    VALID_SOURCE_TAGS,
    DeviceNotFound,
    EPDeviceTarget,
    WinMLDevice,
    WinMLEPNotDiscovered,
    WinMLEPRegistrationFailed,
    _format_bytes,
    auto_detect_device,
    default_ep_for_device,
    ep_short_or_none,
    ep_to_device,
    eps_for_device,
    expand_ep_name,
    resolve_device,
    short_ep_name,
)
from .ep_registry import WinMLEP, WinMLEPDevice, WinMLEPRegistry
from .monitor.ep_monitor import NullEPMonitor, WinMLEPMonitor
from .monitor.hw_monitor import HWMonitor
from .monitor.openvino_monitor import OpenVINOMonitor
from .monitor.qnn_monitor import QNNMonitor
from .monitor.vitisai_monitor import VitisAIMonitor
from .qairt.qairt_session import WinMLQairtSession
from .session import WinMLSession
from .stats import PerfStats


__all__ = [
    "DEVICE_TO_DEVICE_TYPE",
    "DEVICE_TYPE_TO_DEVICE",
    "EP_DEVICE_SPECS",
    "VALID_DEVICES",
    "VALID_EPS",
    "VALID_SOURCE_TAGS",
    "DeviceNotFound",
    "DirectorySource",
    "EPDeviceTarget",
    "EPEntry",
    "HWMonitor",
    "NullEPMonitor",
    "OpenVINOMonitor",
    "PerfStats",
    "QNNMonitor",
    "VitisAIMonitor",
    "WinMLDevice",
    "WinMLEP",
    "WinMLEPDevice",
    "WinMLEPMonitor",
    "WinMLEPNotDiscovered",
    "WinMLEPRegistrationFailed",
    "WinMLEPRegistry",
    "WinMLQairtSession",
    "WinMLSession",
    "auto_detect_device",
    "default_ep_for_device",
    "ep_short_or_none",
    "ep_to_device",
    "eps_for_device",
    "expand_ep_name",
    "resolve_device",
    "short_ep_name",
]

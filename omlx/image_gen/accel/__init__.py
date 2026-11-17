# SPDX-License-Identifier: Apache-2.0
"""Speed optimisations for image-generation models (see docs/image-generation.md)."""

from .a8 import a8_available, a8_status, disable_a8, enable_a8

__all__ = ["a8_available", "a8_status", "disable_a8", "enable_a8"]

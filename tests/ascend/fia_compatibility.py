# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Use the same explicit adapter as normal serving; keep experiment imports."""
from kvcached.integration.vllm.ascend_compat import (  # noqa: F401
    install_causal_window as install,
    status,
)

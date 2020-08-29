#!/usr/bin/env bash
# Usage: at the root dir then: bash format.sh
yapf --in-place --recursive -p --verbose \
--style .style.yapf .
"""
This file contains the helper functions for the chunkenize.py

"""

import bisect
from typing import List, Tuple

import logging

from .quantize_helper import quantize_lrind


def root_handling(
    max_seq_len_in_batch,
    quant_indices,
    how_root,
    context_window_size,
    root_split_offset,
    how_quant,
    quant,
    log=logging.getLogger(__name__),
) -> List[Tuple[int, int]]:
    """
    Handle the root node

    root_split_offset
      Without the offset, appending prompt + context would exceed the max_position_embeddings
    """

    roots = []
    if how_root == "no_split":
        # add the root node
        lrind = (0, max_seq_len_in_batch)
        qlrind = quantize_lrind(lrind, how_quant, quant_indices, quant, log=log)
        roots.append(qlrind)
    elif how_root == "split_max_position_embeddings_no_overlap":
        max_root_len = context_window_size - root_split_offset
        left = 0
        right = min(max_root_len, max_seq_len_in_batch)
        log.debug("max_root_len: %d", max_root_len)
        while left < right and right <= max_seq_len_in_batch:
            lrind = (left, right)
            log.debug("Current lrind: %s", lrind)

            # adjust the right boundary to the quant indices, but smaller than the max_position_embeddings
            qlrind = quantize_lrind(lrind, how_quant, quant_indices, quant, log=log)

            root_len = qlrind[1] - qlrind[0]
            i = 0  # index to adjust the right boundary to further left
            while root_len > max_root_len:
                log.debug(
                    "  Root length %d is greater than the max_position_embeddings %d",
                    root_len,
                    max_root_len,
                )
                qright_index = bisect.bisect_left(quant_indices, right) - 1 - i
                # -1 to find index, such that quant_indicies[qright_index] < right
                log.debug("  New right boundary: %d", quant_indices[qright_index])

                qlrind = (qlrind[0], quant_indices[qright_index])
                root_len = qlrind[1] - qlrind[0]
                i += 1

            assert 0 <= root_len <= max_root_len

            # add the root node
            log.debug("Adding root node: %s", qlrind)
            roots.append(qlrind)

            # update the left and right boundaries
            left = qlrind[1]
            right = min(qlrind[1] + max_root_len, max_seq_len_in_batch)
    else:
        raise ValueError(f"Invalid root handling method {how_root}")

    return roots

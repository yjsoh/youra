import logging
from ..abstract import AbsChunks, AbsChunkIndex


def merge_retrieved(retrieved: list[AbsChunks], log: logging.Logger) -> list[str]:
    if len(retrieved) == 0:
        return []

    sorted_retrieved = sorted(retrieved, key=lambda x: x.index)

    now_lrind = sorted_retrieved[0].lr_index
    # insert first interval into stack
    cleaned = [[now_lrind[0], now_lrind[1]]]
    to_ret = []
    local_merge = [sorted_retrieved[0].passage]
    for r in sorted_retrieved[1:]:
        # Check for overlapping interval,
        # if interval overlap
        log.debug("%s vs now_lrind=%s", r.index, now_lrind)
        if now_lrind[0] <= r.index.left <= now_lrind[1]:
            if r.index.left == now_lrind[1]:
                log.debug("Touching.")
                local_merge.append(r.passage)
            else:
                log.debug("Overlap.")
                # overlap!! Post process overlapping strings
                log.warning(
                    f"Warning! Overlapping chunks detected ({now_lrind} vs {r.index}). TODO merge them."
                )

            cleaned[-1][1] = max(cleaned[-1][1], r.index.right)
            now_lrind = cleaned[-1]

        else:
            log.debug("Isolated. Add local_merge to str_list and resume.")
            cleaned.append([r.index.left, r.index.right])
            now_lrind = r.lr_index
            to_ret.append(" ".join(local_merge))
            local_merge.clear()
            local_merge.append(r.passage)

    if len(local_merge) > 0:
        to_ret.append("".join(local_merge))
        local_merge.clear()

    return to_ret

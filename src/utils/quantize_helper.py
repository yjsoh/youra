"""Helper functions for quantization of the LRIndex text."""

import os
import logging
import bisect
from typing import Tuple

from pymongo import MongoClient
from transformers import AutoTokenizer


def get_or_create_quant_indices(
    tokenizer: AutoTokenizer,
    nlp,
    model_name: str,
    context: str,
    ds_name: str,
    q_index: int,
    force_reconstruct: bool,
    save_quant_indices: bool,
    log: logging.Logger,
) -> list[int]:
    """
    Get or create the quant indices
    """
    quant_indices = get_quant_indices(model_name, ds_name, q_index)
    if force_reconstruct or len(quant_indices) == 0:
        quant_indices = context2sentences(tokenizer, nlp, context, log)

    if save_quant_indices:
        set_quant_indices(model_name, ds_name, q_index, quant_indices)

    return quant_indices


def get_quant_indices(model_name: str, ds_name: str, q_index: int) -> bool:
    """
    Check if the quant indices exists
    """
    mongo_uri = os.getenv("MONGO_URI")
    mongo_client = MongoClient(mongo_uri)
    db = mongo_client["heterollm"]
    collection = db[ds_name]
    query = {
        "q_index": q_index,
        f"{model_name}.sentence_quant_indices": {"$exists": True},
    }

    doc = collection.find(query)
    doc_list = list(doc)
    doc_cnt = len(doc_list)
    if doc_cnt == 0:
        return []
    elif doc_cnt == 1:
        return doc_list[0][f"{model_name}"]["sentence_quant_indices"]
    else:
        raise ValueError("Multiple documents found")


def set_quant_indices(
    model_name: str,
    ds_name: str,
    q_index: int,
    quant_indices: list[int],
    log=logging.getLogger(__name__),
) -> None:
    """
    Save the quant indices
    """
    mongo_uri = os.getenv("MONGO_URI")
    mongo_client = MongoClient(mongo_uri)
    db = mongo_client["heterollm"]
    collection = db[ds_name]
    query = {"q_index": q_index}
    update = {"$set": {f"{model_name}.sentence_quant_indices": quant_indices}}

    result = collection.update_one(query, update, upsert=True)
    log.debug("Upserted: %s", result.upserted_id)


def context2sentences(
    tokenizer,
    nlp,
    ctxt: str,
    self_correction_tolerance: int = 30,
    log=logging.getLogger(__name__),
) -> dict:
    """
    Construct the quant_indicies for the given context.

    Args:
        tokenizer: transformers tokenizer
        nlp: stanza.Pipeline
        ctxt: str, context
        ds_name: str, dataset name
        q_index: int, question index

    Returns:
        quant_indices: List[Tuple[int, str]], list of tuple (lrind, sentence)
        match_count: int, number of matched sentences
        all_count: int, number of all sentences
    """
    encode_options = {"add_special_tokens": False}
    doc = nlp(ctxt)
    sentences = [sentence.text for sentence in doc.sentences]
    return _context2sentences(
        tokenizer,
        sentences,
        ctxt,
        encode_options,
        self_correction_tolerance,
        log,
    )


def targetsentences2sentences(
    tokenizer,
    sentences: list[str],
    ctxt: str,
    self_correction_tolerance: int = 30,
    log=logging.getLogger(__name__),
):
    encode_options = {"add_special_tokens": False}
    return _context2sentences(
        tokenizer,
        sentences,
        ctxt,
        encode_options,
        self_correction_tolerance,
        log,
    )


def _context2sentences(
    tokenizer,
    sentences,
    ctxt: str,
    encode_options,
    self_correction_tolerance: int = 30,
    log=logging.getLogger(__name__),
):
    """Core function to construct the quant_indicies for the given context."""
    target_sentences = []
    decode_sentences = []
    index2sentence = {}
    agg_sentences = ""
    quant_indices = [0]
    log.debug(ctxt)
    log.debug("=" * 100)
    all_ids = tokenizer.encode(ctxt, **encode_options)
    n_tokens = len(all_ids)
    for sentence_index, _sentence in enumerate(sentences):
        if _sentence is None or len(_sentence) == 0:
            continue
        log.debug("-" * 100)
        log.debug("%d %s", sentence_index, _sentence.replace("\n", "<newline>"))

        target_sentence = _sentence
        index2sentence[sentence_index] = target_sentence
        target_sentences.append(target_sentence)

        agg_sentences += target_sentence

        # find the next sentence
        ptr = len(agg_sentences)
        old_ptr = ptr
        while (
            ptr < len(ctxt)
            and sentence_index + 1 < len(sentences)
            and ctxt[ptr] != sentences[sentence_index + 1][0]
        ):
            agg_sentences += ctxt[ptr]
            ptr += 1

        if ptr > old_ptr:
            esc = (
                agg_sentences[-(ptr - old_ptr) :]
                .replace("\n", "<newline>")
                .replace(" ", "_")
            )
        else:
            esc = ""
        log.debug("%d Added %d characters: '%s'", sentence_index, ptr - old_ptr, esc)

        ids = tokenizer.encode(agg_sentences, **encode_options)
        # log.debug("Tokenized: %s", ids)
        log.debug("Target: %s", index2sentence[sentence_index])

        candidate = len(ids)
        log.debug("Initial candidate: %s", candidate)

        # align the target_sentence's beginning to the tokenized sentence
        decoded = tokenizer.decode(
            all_ids[quant_indices[-1] : candidate], skip_special_tokens=True
        )
        decoded = decoded.strip(" \n")
        log.debug("Initial Decoded: %s", decoded)

        target_sentence = _sentence.replace(" .", ".").replace(" ,", ",")
        if target_sentence not in decoded:
            log.debug(
                "Warning: target sentence not in decoded. Cut off the target sentence beginning."
            )
            ptr = 0
            while (
                len(decoded) > 0
                and ptr < len(target_sentence)
                and target_sentence[ptr] != decoded[0]
            ):
                ptr += 1
            if ptr > 0:
                target_sentence = target_sentence[ptr:]
                log.debug("Aligned Target: %s", target_sentence)

        # fuzzy candidate adjustment
        init_candidate = candidate
        visited = set()
        while True:
            decoded = tokenizer.decode(
                all_ids[quant_indices[-1] : candidate], skip_special_tokens=True
            )
            decoded = decoded.strip(" \n\t").strip()
            log.debug("Decoded: %s", decoded)

            if candidate in visited:
                log.debug(
                    "|- Warning: Infinite loop. Revert to init_candidate=%d",
                    init_candidate,
                )
                candidate = init_candidate
                break
            if candidate > n_tokens:
                log.debug(
                    "|- Warning: Out of bound. Revert to init_candidate=%d",
                    init_candidate,
                )
                candidate = init_candidate
                break
            if abs(candidate - len(ids)) > self_correction_tolerance:
                log.debug(
                    "|- Warning: Mismatch self-correction failed. Revert to init_candidate=%d",
                    init_candidate,
                )
                candidate = init_candidate
                break
            visited.add(candidate)

            if decoded != target_sentence:
                log.debug("-" * 20)
                log.debug("|- Warning: Mismatch. Self correcting")

                if target_sentence in decoded.strip():
                    candidate -= 1
                    log.debug(
                        "|- Found target sentence in decoded. Reducing %d", candidate
                    )
                else:
                    candidate += 1
                    log.debug(
                        "|- Not found target sentence in decoded. Increasing %d",
                        candidate,
                    )
            else:
                log.debug("|- Found Matched! %d", candidate)
                break

        decoded = tokenizer.decode(
            all_ids[quant_indices[-1] : candidate], skip_special_tokens=True
        )
        decoded = decoded.strip(" \n\t").strip()
        decode_sentences.append(decoded)

        quant_indices.append(candidate)

    log.debug("=" * 100)
    log.info("Quant Indicies: %s", quant_indices)

    # check if duplicates exist
    if len(quant_indices) != len(set(quant_indices)):
        log.warning("Duplicates exist in quant_indicies")

    # check if the first index is 0
    if quant_indices[0] != 0:
        log.warning("First index is not 0")

    # check if the last index is the end of the text
    if quant_indices[-1] != n_tokens:
        log.warning("Last index is not the end of the text. Manually overwritten.")
        # end of the text, if the -1 is not n_tokens,
        # then this might be because of the trailing spaces.
        quant_indices[-1] = n_tokens

    # validate
    # validate quant_indices are sorted
    assert quant_indices == sorted(quant_indices)

    n_sentence = len(quant_indices) - 1
    log.debug("n_sentence: %d", n_sentence)
    quant_success = 0
    for i, (l, r) in enumerate(zip(quant_indices[:-1], quant_indices[1:])):
        decoded = tokenizer.decode(all_ids[l:r], skip_special_tokens=True)
        decoded = decoded.strip(" \n\t")

        cleaned_sentence = index2sentence[i].replace(" .", ".").replace(" ,", ",")

        if decoded != cleaned_sentence:
            log.warning("-" * 100)
            log.warning("Warning: Mismatch")
            log.warning("Index: %d, l: %d, r: %d", i, l, r)
            log.warning("Target  --------------------")
            log.warning(cleaned_sentence.replace("\n", "<newline>"))
            log.warning("Decoded --------------------")
            log.warning(decoded.replace("\n", "<newline>"))
        else:
            quant_success += 1
    log.info(
        "Matched: %d/%d (%.2f)", quant_success, n_sentence, quant_success / n_sentence
    )

    return {
        "n_tokens": n_tokens,
        "sentence_quant_indices": quant_indices,
        "target_sentences": target_sentences,
        "sentences": decode_sentences,
        "n_sentence": n_sentence,
        "n_sentence_quant_success": quant_success,
        "success_rate": round(100 * quant_success / n_sentence, 2),
    }


def quantize_lrind(
    lrind: Tuple[int, int],
    how: str,
    quant_indices: list[int],
    quant: int,
    log=logging.getLogger(__name__),
) -> Tuple[int, int]:
    """
    Quantize the given lrind

    Return the quantized lrind
    """
    if how == "naive":
        qlrind = _quantize_lrind_naive(lrind, quant)
        if lrind[1] == quant_indices[-1]:
            # if the right index is the end of the text, then the quantized right index should be the end of the text
            qlrind = (qlrind[0], lrind[1])
    elif how == "sentence_enclosing":
        qlrind = _quantize_lrind_sentence_enclosing(lrind, quant_indices)
    else:
        raise ValueError("Invalid quantization method")

    log.debug("Quantize lrind %s: %s -> %s", how, lrind, qlrind)

    return qlrind


def _quantize_lrind_naive(lrind: Tuple[int, int], quant: int) -> Tuple[int, int]:
    """
    Return quantized lrind via naive method

    Naive method: (left // QUANT) * QUANT, (right // QUANT) * QUANT
    """
    left, right = lrind

    qleft = (left // quant) * quant
    qright = (right // quant) * quant

    return (qleft, qright)


def _quantize_lrind_sentence_enclosing(
    lrind: Tuple[int, int],
    quant_indices,
    log=logging.getLogger(__name__),
) -> Tuple[int, int]:
    """
    Return quantized lrind that encloses the given lrind
    """
    n = len(quant_indices)
    left, right = lrind

    # binary search on the quant indices to find index no bigger than the left
    # -1 to find index, such that quant_indicies[qleft] < left
    log.debug(quant_indices)
    log.debug(left)
    li = max(0, bisect.bisect_right(quant_indices, left) - 1)
    # but minimum is 0
    ri = bisect.bisect_left(quant_indices, right)

    assert li <= ri
    assert li >= 0 and li < n
    assert ri >= 0 and ri < n

    qleft = quant_indices[li]
    qright = quant_indices[ri]

    assert qleft <= left
    assert qright >= right

    return (qleft, qright)

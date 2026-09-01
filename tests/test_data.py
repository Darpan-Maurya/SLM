"""Data pipeline tests. The important one is `test_resume_is_bit_exact`."""
import json

import numpy as np
import pytest
import torch

from slm.data import (
    MixtureLoader,
    ResumableLoader,
    Shard,
    ShardWriter,
    TokenDataset,
    read_index,
    write_index,
)
from slm.data.shards import HEADER_BYTES, ShardHeader, dtype_for_vocab


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_dataset(tmp_path, n_tokens=4096, n_shards=4, vocab=1000, name="d"):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    per = n_tokens // n_shards
    metas = []
    tok = 0
    for i in range(n_shards):
        with ShardWriter(str(d / f"shard_{i:04d}.bin"), vocab) as w:
            # ids encode their own global position (mod vocab) so any mis-stitch
            # or off-by-one in the loader shows up as a broken arithmetic run
            w.write(np.arange(tok, tok + per) % vocab)
            tok += per
        metas.append(
            {"name": f"shard_{i:04d}.bin", "n_tokens": per, "n_docs": 1, "dtype": "uint16"}
        )
    write_index(str(d), metas, vocab)
    return str(d)


# --------------------------------------------------------------------------- #
# shard format
# --------------------------------------------------------------------------- #
def test_shard_roundtrip(tmp_path):
    p = str(tmp_path / "s.bin")
    data = np.random.randint(0, 500, size=1234)
    with ShardWriter(p, vocab_size=500) as w:
        w.write(data)
    s = Shard(p)
    assert len(s) == 1234
    assert s.header.vocab_size == 500
    assert np.array_equal(np.asarray(s.tokens), data)


def test_shard_payload_is_page_aligned(tmp_path):
    p = str(tmp_path / "s.bin")
    with ShardWriter(p, 500) as w:
        w.write(np.arange(100))
    assert HEADER_BYTES == 1024
    with open(p, "rb") as f:
        assert ShardHeader.unpack(f.read(HEADER_BYTES)).n_tokens == 100


def test_shard_rejects_out_of_vocab_tokens(tmp_path):
    with pytest.raises(ValueError, match="vocab_size"):
        with ShardWriter(str(tmp_path / "s.bin"), vocab_size=10) as w:
            w.write([1, 2, 11])


def test_shard_is_atomic(tmp_path):
    """A writer that never closes must leave no readable shard behind."""
    p = str(tmp_path / "s.bin")
    w = ShardWriter(p, 100)
    w.write(np.arange(50))
    assert not (tmp_path / "s.bin").exists()
    assert (tmp_path / "s.bin.tmp").exists()
    w.close()
    assert (tmp_path / "s.bin").exists()


def test_dtype_selection():
    assert dtype_for_vocab(32768) == np.uint16
    assert dtype_for_vocab(65536) == np.uint16
    assert dtype_for_vocab(65537) == np.uint32


def test_doc_starts_recorded(tmp_path):
    p = str(tmp_path / "s.bin")
    with ShardWriter(p, 100) as w:
        w.write(np.arange(10))
        w.write(np.arange(20))
        w.write(np.arange(5))
    assert np.array_equal(Shard(p).doc_starts, np.array([0, 10, 30]))


def test_index_json(tmp_path):
    d = make_dataset(tmp_path, n_tokens=800, n_shards=4)
    meta = read_index(d)
    assert meta["n_tokens"] == 800 and meta["n_shards"] == 4


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
def test_dataset_concatenates_shards(tmp_path):
    d = make_dataset(tmp_path, n_tokens=4096, n_shards=4, vocab=100000)
    ds = TokenDataset(d, seq_len=128)
    assert ds.n_tokens == 4096
    assert ds.n_sequences == (4096 - 1) // 128

    for i in (0, 1, 7, ds.n_sequences - 1):
        seq = ds.get(i)
        assert len(seq) == 129
        expected = np.arange(i * 128, i * 128 + 129) % 100000
        assert np.array_equal(seq, expected), f"stitching broke at sequence {i}"


def test_dataset_stitches_across_shard_boundary(tmp_path):
    """A sequence that straddles two shards must be contiguous."""
    d = make_dataset(tmp_path, n_tokens=400, n_shards=4, vocab=100000)  # 100/shard
    ds = TokenDataset(d, seq_len=64)
    seq = ds.get(1)                       # tokens 64..128 -> spans shards 0 and 1
    assert np.array_equal(seq, np.arange(64, 129))


def test_dataset_rejects_too_small_corpus(tmp_path):
    d = make_dataset(tmp_path, n_tokens=64, n_shards=1)
    with pytest.raises(ValueError, match="too few"):
        TokenDataset(d, seq_len=1024)


# --------------------------------------------------------------------------- #
# loader semantics
# --------------------------------------------------------------------------- #
def test_targets_are_inputs_shifted_by_one(tmp_path):
    d = make_dataset(tmp_path, n_tokens=8192, n_shards=2, vocab=100000)
    ds = TokenDataset(d, seq_len=32)
    dl = ResumableLoader(ds, micro_batch_size=4, num_workers=0)
    x, y = dl.next_batch()
    assert x.shape == (4, 32) and y.shape == (4, 32)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_loader_is_deterministic(tmp_path):
    d = make_dataset(tmp_path, n_tokens=16384, n_shards=4)
    ds = TokenDataset(d, seq_len=32)
    a = [dl_x for dl_x, _ in _take(ResumableLoader(ds, 4, seed=5, num_workers=0), 12)]
    b = [dl_x for dl_x, _ in _take(ResumableLoader(ds, 4, seed=5, num_workers=0), 12)]
    for i, (u, v) in enumerate(zip(a, b)):
        assert torch.equal(u, v), f"batch {i} differs across identical loaders"


def test_different_seed_gives_different_order(tmp_path):
    d = make_dataset(tmp_path, n_tokens=16384, n_shards=4)
    ds = TokenDataset(d, seq_len=32)
    a = _take(ResumableLoader(ds, 4, seed=1, num_workers=0), 5)
    b = _take(ResumableLoader(ds, 4, seed=2, num_workers=0), 5)
    assert not all(torch.equal(u[0], v[0]) for u, v in zip(a, b))


def _take(loader, n):
    return [loader.next_batch() for _ in range(n)]


# --------------------------------------------------------------------------- #
# THE resume test
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("workers", [0, 1])
def test_resume_is_bit_exact(tmp_path, workers):
    """Interrupt after 10 batches, restore, and the next 15 must match exactly."""
    d = make_dataset(tmp_path, n_tokens=65536, n_shards=8)
    ds = TokenDataset(d, seq_len=64)

    ref = ResumableLoader(ds, micro_batch_size=4, seed=99, num_workers=workers)
    first = _take(ref, 10)
    state = json.loads(json.dumps(ref.state_dict()))     # force a real serialise
    rest = _take(ref, 15)
    ref.stop()

    fresh = ResumableLoader(ds, micro_batch_size=4, seed=99, num_workers=workers)
    fresh.load_state_dict(state)
    resumed = _take(fresh, 15)
    fresh.stop()

    for i, ((xa, ya), (xb, yb)) in enumerate(zip(rest, resumed)):
        assert torch.equal(xa, xb), f"batch {i} after resume differs"
        assert torch.equal(ya, yb)
    assert first  # the pre-interrupt batches were real


def test_resume_preserves_token_accounting(tmp_path):
    d = make_dataset(tmp_path, n_tokens=65536, n_shards=4)
    ds = TokenDataset(d, seq_len=64)
    dl = ResumableLoader(ds, 4, seed=1, num_workers=0)
    _take(dl, 7)
    assert dl.state.tokens_seen == 7 * 4 * 64
    sd = dl.state_dict()
    dl2 = ResumableLoader(ds, 4, seed=1, num_workers=0)
    dl2.load_state_dict(sd)
    assert dl2.state.tokens_seen == 7 * 4 * 64


def test_resume_refuses_a_changed_seed(tmp_path):
    d = make_dataset(tmp_path, n_tokens=32768, n_shards=2)
    ds = TokenDataset(d, seq_len=64)
    sd = ResumableLoader(ds, 4, seed=1, num_workers=0).state_dict()
    with pytest.raises(ValueError, match="seed"):
        ResumableLoader(ds, 4, seed=2, num_workers=0).load_state_dict(sd)


def test_strict_resume_detects_a_changed_dataset(tmp_path):
    d1 = make_dataset(tmp_path, n_tokens=32768, n_shards=2, name="a")
    d2 = make_dataset(tmp_path, n_tokens=65536, n_shards=2, name="b")
    sd = ResumableLoader(TokenDataset(d1, 64), 4, num_workers=0).state_dict()
    dl = ResumableLoader(TokenDataset(d2, 64), 4, num_workers=0)
    with pytest.raises(ValueError, match="changed size"):
        dl.load_state_dict(sd, strict=True)
    dl.load_state_dict(sd, strict=False)          # opt-out works


# --------------------------------------------------------------------------- #
# distribution across ranks
# --------------------------------------------------------------------------- #
def test_ranks_partition_the_stream_without_overlap(tmp_path):
    # vocab == n_tokens so ids never wrap: the first token of a sequence is a
    # unique fingerprint for it, which is what makes the overlap check meaningful
    d = make_dataset(tmp_path, n_tokens=65536, n_shards=4, vocab=65536)
    ds = TokenDataset(d, seq_len=64)
    loaders = [ResumableLoader(ds, 4, rank=r, world_size=4, seed=7, num_workers=0)
               for r in range(4)]
    seen = []
    for _ in range(5):
        for ld in loaders:
            x, _ = ld.next_batch()
            seen.append(x)
    flat = torch.cat(seen).view(-1, 64)[:, 0].tolist()
    assert len(flat) == len(set(flat)), "two ranks were handed the same sequence"


def test_world_size_change_preserves_the_global_stream(tmp_path):
    """One rank at world=1 must see exactly what two ranks at world=2 see."""
    d = make_dataset(tmp_path, n_tokens=65536, n_shards=4)
    ds = TokenDataset(d, seq_len=64)

    solo = ResumableLoader(ds, micro_batch_size=8, world_size=1, seed=3, num_workers=0)
    pair = [ResumableLoader(ds, micro_batch_size=4, rank=r, world_size=2, seed=3,
                            num_workers=0) for r in range(2)]
    for _ in range(6):
        big = solo.next_batch()[0]
        halves = torch.cat([p.next_batch()[0] for p in pair])
        assert torch.equal(big, halves)


def test_resume_at_a_different_world_size(tmp_path):
    """Checkpoint on 1 GPU, resume on 2, and the job continues the same stream."""
    d = make_dataset(tmp_path, n_tokens=65536, n_shards=4)
    ds = TokenDataset(d, seq_len=64)

    solo = ResumableLoader(ds, micro_batch_size=8, world_size=1, seed=3, num_workers=0)
    _take(solo, 4)
    sd = solo.state_dict()
    expected = [solo.next_batch()[0] for _ in range(3)]

    pair = [ResumableLoader(ds, 4, rank=r, world_size=2, seed=3, num_workers=0)
            for r in range(2)]
    for p in pair:
        p.load_state_dict(sd)
    for want in expected:
        got = torch.cat([p.next_batch()[0] for p in pair])
        assert torch.equal(want, got)


def test_micro_batch_larger_than_corpus_is_rejected(tmp_path):
    d = make_dataset(tmp_path, n_tokens=2048, n_shards=1)
    ds = TokenDataset(d, seq_len=64)
    with pytest.raises(ValueError, match="exceeds"):
        ResumableLoader(ds, micro_batch_size=1024, num_workers=0)


# --------------------------------------------------------------------------- #
# epochs
# --------------------------------------------------------------------------- #
def test_epoch_rolls_over_and_reshuffles(tmp_path):
    d = make_dataset(tmp_path, n_tokens=8192, n_shards=2)
    ds = TokenDataset(d, seq_len=64)              # 127 sequences
    dl = ResumableLoader(ds, micro_batch_size=8, seed=1, num_workers=0)
    spe = dl.steps_per_epoch
    first = [dl.next_batch()[0] for _ in range(spe)]
    assert dl.state.epoch == 0
    second = [dl.next_batch()[0] for _ in range(spe)]
    assert dl.state.epoch == 1
    assert not all(torch.equal(a, b) for a, b in zip(first, second)), \
        "epoch 1 replayed epoch 0's order"


def test_finite_loader_stops(tmp_path):
    d = make_dataset(tmp_path, n_tokens=8192, n_shards=2)
    ds = TokenDataset(d, seq_len=64)
    dl = ResumableLoader(ds, 8, seed=1, num_workers=0, infinite=False)
    n = sum(1 for _ in dl)
    assert n == dl.steps_per_epoch


def test_skip_matches_stepping(tmp_path):
    d = make_dataset(tmp_path, n_tokens=65536, n_shards=2)
    ds = TokenDataset(d, seq_len=64)
    a = ResumableLoader(ds, 4, seed=2, num_workers=0)
    _take(a, 9)
    b = ResumableLoader(ds, 4, seed=2, num_workers=0)
    b.skip(9)
    assert torch.equal(a.next_batch()[0], b.next_batch()[0])


# --------------------------------------------------------------------------- #
# prefetch threads
# --------------------------------------------------------------------------- #
def test_threaded_loader_draws_from_the_same_stream_without_repeats(tmp_path):
    """Multiple workers reorder delivery; they must never duplicate or invent.

    With W workers the consumer sees the first `n` batches to *finish*, which is
    a permutation of some prefix of the reserved order - not necessarily the
    first `n` reserved. So the guarantee to test is: no duplicates, and every
    sequence delivered belongs to the serial stream's leading window.
    """
    d = make_dataset(tmp_path, n_tokens=65536, n_shards=4, vocab=65536)
    ds = TokenDataset(d, seq_len=64)
    n, workers, prefetch = 20, 3, 2
    slack = workers * (prefetch + 1) + workers

    serial = ResumableLoader(ds, 4, seed=8, num_workers=0)
    window = set(
        torch.cat([serial.next_batch()[0] for _ in range(n + slack)])[:, 0].tolist()
    )
    thr = ResumableLoader(ds, 4, seed=8, num_workers=workers, prefetch=prefetch)
    got = torch.cat([thr.next_batch()[0] for _ in range(n)])[:, 0].tolist()
    thr.stop()

    assert len(got) == len(set(got)), "a sequence was delivered twice"
    assert set(got) <= window, "delivered a sequence outside the reserved window"


def test_state_dict_flags_inexactness_with_many_workers(tmp_path):
    d = make_dataset(tmp_path, n_tokens=65536, n_shards=2)
    ds = TokenDataset(d, seq_len=64)
    assert ResumableLoader(ds, 4, num_workers=1).state_dict()["exact"] is True
    assert ResumableLoader(ds, 4, num_workers=4).state_dict()["exact"] is False


# --------------------------------------------------------------------------- #
# mixture
# --------------------------------------------------------------------------- #
def test_mixture_hits_its_target_ratios(tmp_path):
    d1 = make_dataset(tmp_path, n_tokens=32768, n_shards=2, name="web")
    d2 = make_dataset(tmp_path, n_tokens=32768, n_shards=2, name="code")
    m = MixtureLoader(
        {"web": ResumableLoader(TokenDataset(d1, 64), 4, num_workers=0),
         "code": ResumableLoader(TokenDataset(d2, 64), 4, num_workers=0)},
        {"web": 0.8, "code": 0.2},
    )
    got = m.realised_weights(10_000)
    assert abs(got["web"] - 0.8) < 0.01 and abs(got["code"] - 0.2) < 0.01


def test_mixture_resumes(tmp_path):
    d1 = make_dataset(tmp_path, n_tokens=32768, n_shards=2, name="web")
    d2 = make_dataset(tmp_path, n_tokens=32768, n_shards=2, name="code")
    def build():
        return MixtureLoader(
            {"web": ResumableLoader(TokenDataset(d1, 64), 4, seed=4, num_workers=0),
             "code": ResumableLoader(TokenDataset(d2, 64), 4, seed=4, num_workers=0)},
            {"web": 0.7, "code": 0.3},
        )
    a = build()
    _take(a, 11)
    sd = json.loads(json.dumps(a.state_dict()))
    want = [a.next_batch()[0] for _ in range(6)]
    b = build()
    b.load_state_dict(sd)
    for i, w in enumerate(want):
        assert torch.equal(w, b.next_batch()[0]), f"mixture batch {i} differs"

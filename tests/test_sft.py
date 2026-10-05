"""SFT templating, masking, packing, and distillation."""
import json

import pytest
import torch

from slm.config import (
    CheckpointConfig,
    Config,
    DataConfig,
    DistillConfig,
    FinetuneConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
)
from slm.model import Transformer
from slm.sft import ChatTemplate, Message, SFTDataset, read_conversations
from slm.sft.chat import pack_examples, parse_record
from slm.tokenizer import Tokenizer
from slm.train.distill import distillation_loss
from slm.train.distributed import DistInfo
from slm.train.trainer import Trainer

SPECIALS = ("<|endoftext|>", "<|user|>", "<|assistant|>", "<|system|>", "<|pad|>")
CORPUS = "hello world how are you today . i am fine thank you very much . " * 40


@pytest.fixture(scope="module")
def tok():
    return Tokenizer.train([CORPUS], vocab_size=400, special_tokens=SPECIALS,
                           min_frequency=1)


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(path)


# --- templating ----------------------------------------------------------------
def test_loss_mask_covers_only_assistant_tokens(tok):
    t = ChatTemplate(tok)
    ex = t.encode([Message("user", "hello world"), Message("assistant", "i am fine")])
    assert len(ex.ids) == len(ex.mask)
    assert sum(ex.mask) > 0
    # every masked-in position must lie after the assistant marker
    first_on = ex.mask.index(1)
    assert ex.ids[first_on - 1] == tok.special_tokens["<|assistant|>"]
    user_ids = tok.encode_ordinary("hello world")
    for i, tid in enumerate(ex.ids):
        if tid in user_ids and i < first_on:
            assert ex.mask[i] == 0


def test_eos_is_trained_on(tok):
    """The model must learn to stop, so the terminating EOS carries loss."""
    ex = ChatTemplate(tok).encode([Message("user", "hi"), Message("assistant", "yo")])
    assert ex.ids[-1] == tok.eos_id and ex.mask[-1] == 1


def test_multi_turn_masks_every_assistant_turn(tok):
    ex = ChatTemplate(tok).encode([
        Message("user", "one"), Message("assistant", "first reply"),
        Message("user", "two"), Message("assistant", "second reply"),
    ])
    runs = []
    prev = 0
    for m in ex.mask:
        if m == 1 and prev == 0:
            runs.append(1)
        prev = m
    assert len(runs) == 2, "expected one trainable run per assistant turn"


def test_system_prompt_is_not_trained_on(tok):
    ex = ChatTemplate(tok).encode([
        Message("system", "you are helpful"), Message("user", "hi"),
        Message("assistant", "hello"),
    ])
    sys_len = len(tok.encode_ordinary("you are helpful"))
    assert sum(ex.mask[: sys_len + 2]) == 0


def test_render_prompt_ends_ready_for_the_assistant(tok):
    text = ChatTemplate(tok).render_prompt([Message("user", "hi")])
    assert text.endswith("<|assistant|>")


def test_template_requires_chat_special_tokens():
    plain = Tokenizer.train([CORPUS], vocab_size=300, special_tokens=("<|endoftext|>",),
                            min_frequency=1)
    with pytest.raises(ValueError, match="chat special tokens"):
        ChatTemplate(plain)


# --- record parsing ------------------------------------------------------------
@pytest.mark.parametrize("record,expected_roles", [
    ({"messages": [{"role": "user", "content": "a"},
                   {"role": "assistant", "content": "b"}]}, ["user", "assistant"]),
    ({"instruction": "do it", "output": "done"}, ["user", "assistant"]),
    ({"instruction": "do it", "input": "with this", "output": "done"},
     ["user", "assistant"]),
    ({"prompt": "q", "response": "a"}, ["user", "assistant"]),
])
def test_parse_record_formats(record, expected_roles):
    msgs = parse_record(record)
    assert [m.role for m in msgs] == expected_roles


def test_parse_record_rejects_junk():
    assert parse_record({"nothing": "useful"}) is None
    assert parse_record({"instruction": "x"}) is None


def test_read_conversations_skips_bad_lines(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"prompt":"a","response":"b"}\nnot json\n\n{"nope":1}\n')
    assert len(list(read_conversations(str(p)))) == 1


# --- packing -------------------------------------------------------------------
def test_packing_pads_to_a_fixed_width(tok):
    t = ChatTemplate(tok)
    ex = [t.encode([Message("user", "hi"), Message("assistant", "there")])
          for _ in range(9)]
    x, m = pack_examples(ex, seq_len=48, pad_id=0)
    assert x.shape[1] == 49 and m.shape == x.shape
    assert set(m.flatten().tolist()) <= {0, 1}


def test_packing_never_masks_padding_in(tok):
    t = ChatTemplate(tok)
    ex = t.encode([Message("user", "hi"), Message("assistant", "yo")])
    x, m = pack_examples([ex], seq_len=64, pad_id=0)
    n = len(ex.ids)
    assert m[0][n:].sum() == 0, "padding was marked trainable"
    assert x[0][n:].tolist() == [0] * (65 - n)


def test_overlong_example_is_truncated_or_dropped(tok):
    t = ChatTemplate(tok)
    long_ex = t.encode([Message("user", "hi"), Message("assistant", "word " * 500)])
    kept, _ = pack_examples([long_ex], seq_len=32, pad_id=0)
    assert kept.shape[1] == 33
    dropped, _ = pack_examples([long_ex], seq_len=32, pad_id=0, drop_long=True)
    assert len(dropped) == 0


def test_sft_dataset_reports_trainable_fraction(tmp_path, tok):
    path = write_jsonl(tmp_path / "d.jsonl",
                       [{"prompt": "hello world", "response": "i am fine"}] * 20)
    ds = SFTDataset(path, tok, seq_len=64)
    assert 0.0 < ds.trainable_fraction < 1.0
    assert ds.n_examples == 20
    assert "trainable" in repr(ds)


def test_sft_dataset_rejects_an_empty_file(tmp_path, tok):
    path = write_jsonl(tmp_path / "e.jsonl", [{"junk": 1}])
    with pytest.raises(ValueError, match="no usable conversations"):
        SFTDataset(path, tok, seq_len=32)


# --- distillation --------------------------------------------------------------
def test_distillation_loss_is_zero_when_models_agree():
    logits = torch.randn(2, 5, 20)
    assert distillation_loss(logits, logits.clone(), temperature=2.0).abs() < 1e-5


def test_distillation_loss_grows_with_disagreement():
    a = torch.randn(2, 5, 20)
    near = a + 0.05 * torch.randn_like(a)
    far = torch.randn_like(a) * 5
    assert distillation_loss(a, near) < distillation_loss(a, far)


def test_distillation_respects_the_loss_mask():
    s = torch.randn(1, 4, 10)
    t = torch.randn(1, 4, 10)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    masked = distillation_loss(s, t, loss_mask=mask)
    manual = distillation_loss(s[:, :2], t[:, :2])
    assert torch.allclose(masked, manual, atol=1e-5)


def test_topk_distillation_converges_to_the_full_kl():
    """Truncating the teacher is an approximation that must improve with k."""
    torch.manual_seed(0)
    s = torch.randn(2, 6, 100)
    t = torch.randn(2, 6, 100) * 3
    full = float(distillation_loss(s, t, top_k=0))
    errs = [abs(float(distillation_loss(s, t, top_k=k)) - full) for k in (5, 25, 99)]
    assert errs[0] > errs[1] > errs[2], errs


# --- end to end ----------------------------------------------------------------
def sft_config(tmp_path, data_path, tokenizer_path, steps=12):
    return Config(
        model=ModelConfig(vocab_size=400, n_layer=2, n_head=2, n_kv_head=1,
                          d_model=64, max_seq_len=64, ffn_hidden=96,
                          nope_every=0, doc_masking=False, mtp_depth=0),
        data=DataConfig(seq_len=64, num_workers=0),
        optim=OptimConfig(kind="adamw", lr=3e-3, warmup_steps=2, schedule="constant",
                          fused=False, zloss=0.0),
        train=TrainConfig(run_name="sft", out_dir=str(tmp_path / "runs"),
                          max_steps=steps, global_batch_tokens=256,
                          micro_batch_size=4, dtype="float32", compile=False,
                          log_every=1, eval_every=0),
        ckpt=CheckpointConfig(interval=0, async_save=False, permanent_every=0),
        finetune=FinetuneConfig(mode="sft", data=data_path, tokenizer=tokenizer_path),
    )


def test_sft_run_trains_and_lowers_loss(tmp_path, tok):
    tok_path = str(tmp_path / "tok.json")
    tok.save(tok_path)
    data = write_jsonl(tmp_path / "sft.jsonl",
                       [{"prompt": "hello world", "response": "i am fine thank you"}] * 60)
    cfg = sft_config(tmp_path, data, tok_path, steps=25)
    t = Trainer(cfg, dist_info=DistInfo(device=torch.device("cpu")))
    t.train()

    import json as _json
    losses = []
    with open(f"{t.run_dir}/logs/metrics.jsonl") as f:
        for line in f:
            rec = _json.loads(line)
            if "train/loss" in rec:
                losses.append(rec["train/loss"])
    assert losses[-1] < losses[0] - 0.5, losses[:3] + losses[-3:]


def test_distillation_run_executes(tmp_path, tok):
    tok_path = str(tmp_path / "tok.json")
    tok.save(tok_path)
    teacher_cfg = ModelConfig(vocab_size=400, n_layer=2, n_head=2, n_kv_head=1,
                              d_model=64, max_seq_len=64, ffn_hidden=96,
                              nope_every=0, doc_masking=False)
    teacher = Transformer(teacher_cfg)
    tdir = tmp_path / "teacher"
    tdir.mkdir()
    torch.save(teacher.state_dict(), tdir / "model.pt")
    with open(tdir / "config.json", "w") as f:
        json.dump({"model": {k: v for k, v in teacher_cfg.__dict__.items()}}, f)

    data = write_jsonl(tmp_path / "sft.jsonl",
                       [{"prompt": "hello", "response": "i am fine"}] * 40)
    cfg = sft_config(tmp_path, data, tok_path, steps=6)
    cfg.distill = DistillConfig(teacher=str(tdir), alpha=0.5, temperature=2.0)
    t = Trainer(cfg, dist_info=DistInfo(device=torch.device("cpu")))
    assert t.teacher is not None
    assert t.train() == 0

"""Which models the hardware can hold, and at what precision.

A benchmark that silently skips the models that did not fit tells you the wrong
thing: the ones missing from the table are usually the ones you most wanted to
know about. So every model is planned, and each is placed in the highest
precision the available cards can actually hold.

Three tiers come out of that:

    A  native      fits unquantized -- the number means what it says
    B  quantized   only fits with weights compressed; the score includes
                   whatever that cost, which is itself worth measuring
    C  deferred    does not fit here at any useful precision. Planned, not
                   run, and carried in the results as a gap rather than an
                   omission -- run it on hardware that can hold it and the
                   difference against tier B is the quantization penalty.

Sizes are computed from parameter counts rather than measured, because the
decision has to be made *before* the weights are downloaded. They are estimates
and the runner reports what the model actually used.
"""
from dataclasses import dataclass

# Bytes per parameter. int8 and nf4 are bitsandbytes' two options; nf4 is the
# 4-bit normal-float that keeps more of a normally-distributed weight than a
# plain int4 would.
BYTES_PER_PARAM = {"fp16": 2.0, "int8": 1.0, "nf4": 0.5}

# Weights are not the whole cost: activations, the KV cache and the processor's
# buffers all live on the card too. 15% is a working figure, not a guarantee.
RUNTIME_OVERHEAD = 1.15

# Tried in order, so a model lands in the highest precision that fits and on
# the fewest cards. One card is preferred over two at equal precision: it
# leaves the other free, and sharding pays a PCIe cost on every forward pass.
PLACEMENT_LADDER = [
    ("fp16", 1), ("fp16", 2),
    ("int8", 1), ("int8", 2),
    ("nf4", 1), ("nf4", 2),
]

TIER_OF_PRECISION = {"fp16": "A", "int8": "B", "nf4": "B"}


@dataclass(frozen=True)
class Placement:
    """How a model can be loaded here, or why it cannot."""
    model: str
    params_b: float
    precision: str | None      # None when nothing fits
    cards: int
    vram_gb: float             # estimated, at the chosen precision
    native_vram_gb: float      # what it would need unquantized
    tier: str                  # A | B | C

    @property
    def runnable(self) -> bool:
        return self.tier != "C"

    @property
    def quantized(self) -> bool:
        return self.tier == "B"

    def describe(self) -> str:
        if not self.runnable:
            return f"deferred — needs {self.native_vram_gb:.0f} GB unquantized"
        where = f"{self.cards} card" + ("s" if self.cards > 1 else "")
        return f"{self.precision} on {where}, ~{self.vram_gb:.1f} GB"


def estimate_vram(params_b: float, precision: str) -> float:
    """Rough resident size of a model's weights plus its runtime overhead."""
    return params_b * BYTES_PER_PARAM[precision] * RUNTIME_OVERHEAD


def place(model: str, params_b: float, usable_gb_per_card: float,
          cards: int = 2, allow_quantization: bool = True) -> Placement:
    """The highest-precision placement that fits, or tier C.

    `allow_quantization=False` forces the native answer, which is how a tier-C
    run is planned for hardware that can hold it: same model, no compression,
    directly comparable with the tier-B result.
    """
    native = estimate_vram(params_b, "fp16")
    for precision, needed_cards in PLACEMENT_LADDER:
        if needed_cards > cards:
            continue
        if not allow_quantization and precision != "fp16":
            break
        size = estimate_vram(params_b, precision)
        if size <= usable_gb_per_card * needed_cards:
            return Placement(model=model, params_b=params_b, precision=precision,
                             cards=needed_cards, vram_gb=size, native_vram_gb=native,
                             tier=TIER_OF_PRECISION[precision])
    return Placement(model=model, params_b=params_b, precision=None, cards=0,
                     vram_gb=0.0, native_vram_gb=native, tier="C")


def plan_placements(models: dict[str, float], usable_gb_per_card: float,
                    cards: int = 2) -> list[Placement]:
    """Place every model, then add a tier-C entry for each one that had to be
    quantized to fit.

    That second pass is the point of the tier split. A quantized 32B and a
    native 8B on the same table differ in two ways at once, and the table
    cannot say which one moved the score. The deferred native run is the
    control that separates them.
    """
    placements = [place(name, params, usable_gb_per_card, cards)
                  for name, params in models.items()]
    deferred = [place(p.model, p.params_b, usable_gb_per_card, cards,
                      allow_quantization=False)
                for p in placements if p.quantized]
    return placements + deferred


def usable_vram(reserve_gb: float = 1.0) -> tuple[float, int]:
    """(usable GB per card, card count) on this machine.

    The reserve is for the CUDA context and allocator fragmentation, which are
    real and are not in any parameter count.
    """
    try:
        import torch
    except ImportError:
        return 0.0, 0
    if not torch.cuda.is_available():
        return 0.0, 0
    cards = torch.cuda.device_count()
    smallest = min(torch.cuda.get_device_properties(i).total_memory
                   for i in range(cards)) / 1024 ** 3
    return max(smallest - reserve_gb, 0.0), cards

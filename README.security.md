# Transformers 4.46 security backports

This fork starts at v4.46.3, commit
`052e652d6d53c2b26ffde87e039b723949a53493`.
The package version is `4.46.3+security.1`. It retains the 4.46 API used by
Splattie's Puppeteer integration.

| Advisory | Change |
| --- | --- |
| GHSA-69w3-r845-3855 | Restricted Trainer model, optimizer, scheduler and RNG checkpoint readers. |
| GHSA-4w7r-h757-3r74 | Preserve AdamWeightDecay regex patterns with an explicit execution timeout. |
| GHSA-rcv9-qm8p-9p6j | Preserve CLVP number matching with an explicit execution timeout. |
| GHSA-59p9-h35m-wg4g | Scan Marian language prefixes without backtracking. |
| GHSA-9356-575x-2w9m | Bound TF weight names before separator normalization. |
| GHSA-37mw-44qp-f5jm | Scan Donut start tags directly. |
| GHSA-jjph-296x-mrcr | Parse dynamic-module imports with Python's AST. |
| GHSA-q2wp-rjmx-x6x9 | Select configuration filenames without a backtracking regex. |
| GHSA-qq3j-4f4f-9583 | Bound doctest code-block matching. |
| GHSA-fpwr-67px-3qhx | Bound Japanese tokenizer matching. |
| GHSA-6rvg-6v2m-4j46 | Simplify Nougat list normalization and bound heading matching. |
| GHSA-wrfc-pvp9-mr9g | Restrict NumPy pickle globals and Torch loads in MaskFormer conversion. |
| GHSA-hxxf-235m-72v3 | Restrict NumPy pickle globals in Reformer conversion. |
| GHSA-qxrp-vhvm-j765 | Safely parse MobileViT YAML and restrict Torch checkpoint loading. |

The following affected features do not exist in this baseline:
named chat-template file export (GHSA-xrqw-3rrv-vx5w), LightGlue
(GHSA-fgcw-684q-jj6r), Hub attention kernels (GHSA-29pf-2h5f-8g72), and
YouTube video URL loading (GHSA-phhr-52qp-3mj4).

Upstream changes used as references:
`03c8082ba4594c9b8d6fe190ca9bed0e5f8ca396`,
`d37f7517972f67e3f2194c000ed0f87f064e5099`,
`54a02160eb030da9be18231c77791f2eb3a52216`,
`701caef704e356dc2f9331cc3fd5df0eccb4720a`,
`ebbe9b12dd75b69f92100d684c47f923ee262a93`,
`126abe3461762e5fc180e7e614391d1b4ab051ca`,
`8cb522b4190bd556ce51be04942720650b1a3e57`,
`92c5ca9dd70de3ade2af2eb835c96215cc50e815`,
`deac971c469bcbb182c2e52da0b82fb3bf54cccf`, and
`e0ae9b59747445d6e470e04dc3ed45128123ee4d`.

Checkpoint conversion remains available. NumPy checkpoints permit only the
array reconstruction globals required by supported protocols and reject pickle
extension and persistent-reference opcodes. Native Detectron2 checkpoints use
the restricted reader even with released Detectron2 versions whose parent
reader bypasses the Torch-loading hook.

Torch checkpoint loading requires a stable Torch version at least 2.6.0 or
the exact maintained builds `2.3.0+cu121.security1` (CUDA wheel) and
`2.3.0+security.1` (native source), with their restricted-loader APIs.
Unpatched builds and prereleases are rejected. RNG restoration validates
generator state before changing process state and scopes NumPy permissions
to the current load.

The CPU security workflow uses Torch 2.6 as a test harness. Splattie retains
its CUDA 12.1 stack. Compatibility with the compiled Torch 2.3 backport and GPU
generation must be validated separately.

# Third-party notices

This project vendors code from other open-source projects. Their licenses and
copyright notices are reproduced below, as required.

## nanochat

Files derived from nanochat (https://github.com/karpathy/nanochat), vendored
because nanochat is not installable under this project's torch pin (ADR 0011):

- `cankar/model/gpt.py` - the GPT architecture (ported, forward path verbatim; see ADR 0016)
- `cankar/model/flash_attention.py` - the FA3/SDPA switch (ported)
- `cankar/evals/vendored_bpb.py` - the bits-per-byte metric
- `cankar/tokenizer/vendored.py` - tokenizer constants (special tokens, split pattern)

Vendored at nanochat commit `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`.

```
MIT License

Copyright (c) 2025 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

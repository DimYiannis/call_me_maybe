1. Prompt
2. Tokenization
3. Input IDs
4. LLM Processing
5. Logits                     ← decoder gets these
6. identify valid tokens       ← constraints + vocabulary
7. mask invalid to -inf        ← decoder uses serializer output
8. Token Selection             ← decoder picks argmax
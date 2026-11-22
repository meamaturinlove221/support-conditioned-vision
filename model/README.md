# Local model inputs

No weights are distributed here. Generic inference uses a separately obtained official DINOv2 backbone and local Support and Query images. The optional custom adapter requires a compatible, authorized local artifact. Use trusted local checkpoints only. The adapter loader restricts deserialization to weights-only mode and does not fall back to unrestricted loading. Legacy files that require arbitrary Python objects are not supported.
 The artifact must contain a compatible eight-tensor `adapter` state dictionary.

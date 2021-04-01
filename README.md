# PyTorch Wrapper for TensorFlow

This is a proof-of-concept script that wraps a TensorFlow model in PyTorch autograd function.

Forwarding works as expected.
Backwarding can pass through TF model, but we don't save gradients of parameters in TF.
So Training TF model in PyTorch is not supported. Also, we cannot store intermediate results in TF model, so when backwarding, an extra forward pass is done.

Tested with TensorFlow 1.14.0, PyTorch 1.8.1.

See `tests/test_tf_wrapper.py` for example usage.

# TODO

* Test with TensorFlow 2, and avoid the extra forward pass.
* We may use dlpack to convert PyTorch CUDA tensor to TensorFlow 2 eager tensor to avoid extra copy.

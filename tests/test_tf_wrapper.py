import tensorflow as tf
import torch
import torch.cuda

from tf_wrapper import tf_wrapper

def test():
    tf.compat.v1.disable_eager_execution()
    x = tf.placeholder(tf.float32, shape=(4,), name='x')
    y = 2 * x

    opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=opts))

    tf_module = tf_wrapper(sess, y, [x])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = torch.ones(4, device=device, requires_grad=True)
    out = tf_module(data)
    print(repr(out)) # [2., 2., 2., 2.]

    w = torch.arange(0,4,1, dtype=torch.float32, device=device)
    (out * w).sum().backward()
    print(repr(data.grad)) # [0., 2., 4., 6.]

if __name__ == '__main__':
    test()

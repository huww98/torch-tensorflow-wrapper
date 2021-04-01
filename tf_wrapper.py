import tensorflow as tf
import torch
import torch.autograd


def tf_wrapper(session: tf.Session, fetches, feeds, output_device=None):
    if isinstance(fetches, tf.Tensor):
        single_fetch = True
        fetches = [fetches]
    else:
        single_fetch = False
        fetches = fetches

    with fetches[0].graph.as_default():
        with tf.name_scope('grad_ys'):
            grad_tensor = [tf.placeholder(dtype=tf.float32, shape=o.shape) for o in fetches]
        dx = tf.gradients(fetches, feeds, grad_ys=grad_tensor)


    class TFModule(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args, **kwargs):
            feed_dict = {
                **{feeds[i]: a for i, a in enumerate(args)},
                **{feeds[k]: a for k, a in kwargs.items()},
            }

            ctx.in_device = next(iter(feed_dict.values())).device
            if output_device is None:
                device = ctx.in_device
            else:
                device = output_device

            for k, v in feed_dict.items():
                feed_dict[k] = v.cpu()

            ctx.save_for_backward(*feed_dict.values())
            ctx.feeds = feed_dict

            out = session.run(fetches, feed_dict=feed_dict)
            out = [torch.from_numpy(i).to(device) for i in out]
            if single_fetch:
                out = out[0]
            return out

        @staticmethod
        def backward(ctx, *grad_outputs):
            back_feeds = {
                **ctx.feeds,
                **{t: g.cpu() for t, g in zip(grad_tensor, grad_outputs)},
            }
            grad_in = session.run(dx, feed_dict=back_feeds)
            grad_in = tuple(torch.from_numpy(i).to(ctx.in_device) for i in grad_in)
            return grad_in

    return TFModule.apply

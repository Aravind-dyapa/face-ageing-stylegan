
import torch
import torch.nn.functional as F


def upfirdn2d_native(
    input,
    kernel,
    up_x,
    up_y,
    down_x,
    down_y,
    pad_x0,
    pad_x1,
    pad_y0,
    pad_y1
):
    batch, channel, in_h, in_w = input.shape

    # Insert zeros for upsampling.
    if up_x > 1 or up_y > 1:
        out = input.new_zeros(
            batch, channel, in_h * up_y, in_w * up_x
        )
        out[:, :, ::up_y, ::up_x] = input
    else:
        out = input

    # Positive padding.
    out = F.pad(
        out,
        (
            max(pad_x0, 0),
            max(pad_x1, 0),
            max(pad_y0, 0),
            max(pad_y1, 0)
        )
    )

    # Negative padding means cropping.
    if pad_y0 < 0:
        out = out[:, :, -pad_y0:, :]

    if pad_y1 < 0:
        out = out[:, :, :out.shape[2] + pad_y1, :]

    if pad_x0 < 0:
        out = out[:, :, :, -pad_x0:]

    if pad_x1 < 0:
        out = out[:, :, :, :out.shape[3] + pad_x1]

    # Apply the filtering kernel independently to each channel.
    kernel = kernel.to(device=input.device, dtype=input.dtype)
    kernel = torch.flip(kernel, [0, 1])
    kernel = kernel.view(1, 1, kernel.shape[0], kernel.shape[1])
    kernel = kernel.repeat(channel, 1, 1, 1)

    out = F.conv2d(
        out,
        kernel,
        groups=channel,
        stride=(down_y, down_x)
    )

    return out


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    return upfirdn2d_native(
        input,
        kernel,
        up,
        up,
        down,
        down,
        pad[0],
        pad[1],
        pad[0],
        pad[1]
    )

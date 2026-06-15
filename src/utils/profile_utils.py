def count_parameters(model):
    trainable = 0
    non_trainable = 0

    for parameter in model.parameters():
        count = parameter.numel()
        if parameter.requires_grad:
            trainable += count
        else:
            non_trainable += count

    return {
        "trainable": trainable,
        "non_trainable": non_trainable,
        "total": trainable + non_trainable,
    }


def format_number(value):
    return f"{int(value):,}"

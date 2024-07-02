import torch


def binary_inference(model, x, y, loss_fn, correct, total):
    logits = model(x).logits
    probs = torch.sigmoid(logits)
    predicted = probs.round()
    loss = loss_fn(logits, y)
    correct += (predicted == y).sum().item()
    total += y.numel()

    # Convert one-hot encoded predictions to binary form
    binary_predictions = torch.argmax(predicted, dim=-1).cpu()
    return binary_predictions, loss, correct, total


def multitoken_inference(
    model, x, y, target_bits, loss_fn, correct, total, config, tokenizer
):
    # Model(x).logits has shape [batch_size, sequence_length, 2**target_bits]
    logits = model(x).logits

    _, target_indices = y.max(dim=2)
    target = target_indices.view(-1)

    # Flatten logits and target for loss calculation
    flattened_logits = logits.view(-1, 2**target_bits)
    loss = loss_fn(flattened_logits, target)

    if config["evaluate_all_bits"]:
        predicted = flattened_logits.argmax(dim=1)
        correct += (predicted == target).sum().item()
        total += target.size(0)
        binary_predictions = tokenizer.detokenize(predicted.cpu().tolist())
    else:
        # Evaluate only the last token's logits for each batch item
        last_logits = logits[:, -1, :]  # Shape: [batch_size, 2**target_bits]
        last_predicted = last_logits.argmax(dim=1)  # Predictions for the last token

        # Corresponding targets for the last token
        last_target = y[:, -1, :]

        correct += (last_predicted == last_target).sum().item()
        total += x.size(0)  # Total number of examples in the batch

        binary_predictions = tokenizer.detokenize(last_predicted.cpu().tolist())

    binary_predictions = torch.tensor(
        [int(bit) for bit in binary_predictions], dtype=torch.int32
    )

    return binary_predictions, loss, correct, total


def multitoken_inference(
    model, x, y, target_bits, loss_fn, correct, total, config, tokenizer
):
    # model(x).logits has shape [batch_size, sequence_length, 2**target_bits]
    logits = model(x).logits

    _, target_indices = y.max(dim=2)
    target = target_indices.view(-1)

    # Flatten logits and target for loss calculation
    flattened_logits = logits.view(-1, 2**target_bits)
    loss = loss_fn(flattened_logits, target)

    if config["evaluate_all_bits"]:
        predicted = flattened_logits.argmax(dim=1)
        correct += (predicted == target).sum().item()
        total += target.size(0)
        binary_predictions = tokenizer.detokenize(predicted.cpu().tolist())
    else:
        # Evaluate only the last token's logits for each batch item
        last_logits = logits[:, -1, :]  # Shape: [batch_size, 2**target_bits]

        last_predicted = last_logits.argmax(dim=1)  # Predictions for the last token

        last_target = y[:, -1, :].max(dim=1)[
            1
        ]  # Get the index of the max logit for the last token

        correct += (last_predicted == last_target).sum().item()
        total += x.size(0)  # Total number of examples in the batch

        binary_predictions = tokenizer.detokenize(last_predicted.cpu().tolist())

    binary_predictions = torch.tensor(
        [int(bit) for bit in binary_predictions], dtype=torch.int32
    )

    return binary_predictions, loss, correct, total


def eval_multi(predictions, target, target_bits, correct, total, config):
    if config["evaluate_all_bits"]:
        # This computes a tensor where each element is 1 if all corresponding labels match, and 0 otherwise
        correct_predictions = (predictions == target).all(dim=2)
        # Sum up the correct predictions to get the total number of completely correct samples
        correct += correct_predictions.sum().item()
        # The total number of samples is just the size of the first dimension of target tensor
        total += target.size(0) * target.size(1)
    else:
        predictions_last_bits = predictions[:, -1, -target_bits:]
        target_last_bits = target[:, -1, -target_bits:]

        # Compute correct predictions for the last target_bits bits
        correct_predictions_last_bits = (predictions_last_bits == target_last_bits).all(
            dim=1
        )

        # Sum up the correct predictions
        correct += correct_predictions_last_bits.sum().item()

        # The total number of samples is the first dimension of  target tensor
        total += target.size(0)

    binary_predictions = predictions.view(-1)
    return binary_predictions, correct, total


def autoregressive_inference(
    model, x, y, target_bits, loss_fn, correct, total, config, device
):
    target = y.float()
    x_current = x.to(device)
    predictions_sequence = []

    for step in range(target_bits):
        logits = model(x_current).logits
        probs = torch.sigmoid(logits)
        predicted_bits = torch.argmax(probs, dim=-1)

        # Remove the first element from x_current and append the new predicted element
        x_current = torch.cat([x_current[:, 1:], predicted_bits[:, -1:]], dim=1)

        # Expand dimensions of predicted_bits to make it compatible for concatenation
        predicted_bits_expanded = predicted_bits.unsqueeze(-1)

        predictions_sequence.append(predicted_bits_expanded)

    concatenated_bits = torch.cat(predictions_sequence, dim=-1)
    # we need to one-hot encode the predictions
    # Converting each sequence of bits to a decimal number. We multiply each bit by its corresponding power of 2
    decimal_predictions = torch.sum(
        concatenated_bits
        * 2
        ** torch.arange(target_bits - 1, -1, -1, device=device)
        .unsqueeze(0)
        .unsqueeze(0),
        dim=-1,
    )
    # Converting each decimal number to a one-hot encoded tensor
    predictions = torch.nn.functional.one_hot(
        decimal_predictions, num_classes=2**target_bits
    ).to(torch.float)
    loss = loss_fn(predictions, target)

    binary_predictions, correct, total = eval_multi(
        predictions, target, target_bits, config
    )

    return binary_predictions, loss, correct, total


if __name__ == "__main__":
    print(
        "This is a module with the implementation of the inference mechanisms for the GPT-2 model."
    )

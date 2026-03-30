import torch


def greedy_predict(model, features, vocab, max_len=50, device="cpu"):
    """Generates a caption using greedy (argmax) decoding."""
    model.eval()
    with torch.no_grad():
        features = model.encoder(features.to(device))
        inputs = features.unsqueeze(1)
        states = None
        sampled_ids = []

        for _ in range(max_len):
            hiddens, states = model.decoder.lstm(inputs, states)
            outputs = model.decoder.linear(hiddens.squeeze(1))
            predicted = outputs.argmax(1)
            sampled_ids.append(predicted.item())

            if vocab.itos[predicted.item()] == "<end>":
                break

            inputs = model.decoder.embed(predicted).unsqueeze(1)

    caption = vocab.decode(sampled_ids)
    if "<end>" in caption:
        caption = caption[: caption.index("<end>")]
    return " ".join(caption)

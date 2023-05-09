import torch
from BaseTransformer import BaseTransformer
from Data.vocab import Vocab
from Utils import tokenizing


def greedy_decode(input_text="", 
                  model=None, 
                  src_vocab=None, 
                  tgt_vocab=None, 
                  max_output_length=50):
    assert model!=None, "Need trained model"
    assert src_vocab!=None, "Need source vocab"
    assert tgt_vocab!=None, "Need target vocab"

    input_tokens = [src_vocab[i] for i in tokenizing(input_text)]
    encoder_input = torch.Tensor([input_tokens])
    model.eval()
    with torch.no_grad():
        encoder_output = model.encode(encoder_input)

    decoder_input = torch.Tensor([[tgt_vocab.BOS]]).long()
    max_output_length = encoder_input.shape[-1] + 50
    # Autoregressive
    for _ in range(max_output_length):
        # Decoder prediction
        logits = model.decode(encoder_output, decoder_input)

        # Greedy selection
        token_index = torch.argmax(logits[:, -1], keepdim=True)
        
        # EOS is most probable => Exit
        if token_index.item()==tgt_vocab.EOS:
            break

        # Next Input to Decoder
        decoder_input = torch.cat([decoder_input, token_index], dim=1)
    decoder_output = decoder_input[0, 1:].numpy()
    return [tgt_vocab[o] for o in decoder_output]
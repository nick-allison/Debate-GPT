<<<<<<< HEAD
=======
#!/usr/bin/env python3

>>>>>>> 7d5b632a9bd1c8ce37d6e30c350fbe1b062b9b2d
import argparse
import sys
import os
import time
import torch
import torch.nn.functional as F

from models import GPTLanguageModel
<<<<<<< HEAD
from tokenizers import BPETokenizer
=======
from tokenizers import CharacterTokenizer
>>>>>>> 7d5b632a9bd1c8ce37d6e30c350fbe1b062b9b2d
from dataset import TextDataset
from train import train_loop, estimate_loss

def indefinite_continuous_generate(model, tokenizer, prompt):
    """
<<<<<<< HEAD
    Generate tokens indefinitely. Done one token at a time
=======
    Generate tokens indefinitely. We do it one token at a time
>>>>>>> 7d5b632a9bd1c8ce37d6e30c350fbe1b062b9b2d
    until user presses Ctrl+C.
    """
    device = next(model.parameters()).device
    if prompt.strip():
        start_ids = tokenizer.encode(prompt)
        context_ids = torch.tensor([start_ids], dtype=torch.long, device=device)
    else:
        context_ids = torch.zeros((1,1), dtype=torch.long, device=device)

    model.eval()
    try:
        while True:
            idx_cond = context_ids[:, -model.context_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
<<<<<<< HEAD
            #print("\n\nToken:", next_id, "\n\n")
=======
>>>>>>> 7d5b632a9bd1c8ce37d6e30c350fbe1b062b9b2d
            context_ids = torch.cat([context_ids, next_id], dim=1)

            new_token_str = tokenizer.decode([next_id.item()])
            print(new_token_str, end="", flush=True)
            time.sleep(0.03)  # small delay so it's not super fast
    except KeyboardInterrupt:
        print("\n[Stopped continuous generation by Ctrl+C]")

<<<<<<< HEAD
def debate_mode(model, tokenizer):
    """
    Interactively chat: user inputs a line, model responds.
    The user message is augmented with a newline and the HERO: token.
    The model generates tokens one-by-one until it predicts
    any of the stopping tokens:
      - HERO: (500)
      - VILLIAN: (501)
      - <|END_HERO|> (503)
      - <|SEP|> (504)
    503 is the actual stopping points, but if it is predicting one of the others(500, 501, 504), then this is naturally
    a reasonable time to stop too based on how the fine tune data was set up.
    Additionally, any generated MOD: token (502) is replaced with "MODERATOR:\n".
=======

def debate_mode(model, tokenizer):
    """
    Interactively chat: user inputs a line, model responds.
    We stop the model response if we see a newline token or a special token.
>>>>>>> 7d5b632a9bd1c8ce37d6e30c350fbe1b062b9b2d
    """
    device = next(model.parameters()).device
    conversation = ""
    model.eval()

    print("Debate mode. Type 'exit' to quit.\n")

    while True:
        user_msg = input("YOU: ")
        if user_msg.strip().lower() == "exit":
            print("Bye!")
            break

<<<<<<< HEAD
        # Append the user message, a newline, and the HERO: token.
        conversation += "VILLIAN: " + user_msg + "\nHERO: "

        # Encode the entire conversation so far.
        context_ids = torch.tensor([tokenizer.encode(conversation)], dtype=torch.long, device=device)
        generated_ids = context_ids

        # Container for the model-generated response (for newline counting)
        model_response = ""

        print("MODEL: ", end="", flush=True)

        while True:
            idx_cond = generated_ids[:, -model.context_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            new_token_id = next_token.item()

            # Stop generation if the token is HERO: (500), VILLIAN: (501),
            # <|END_HERO|> (503), or <|SEP|> (504).
            if new_token_id in [500, 501, 503, 504]:
                break

            # Replace MOD: (502) with "MODERATOR:\n"
            if new_token_id == 502:
                token_str = "MODERATOR:\n"
            else:
                token_str = tokenizer.decode([new_token_id])

            # Append the new token to the conversation and the model response.
            conversation += token_str
            model_response += token_str
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Print the token immediately.
            print(token_str, end="", flush=True)
        
        #Newline before the next prompt for input
        print()

=======
        # Append user message
        conversation += "USER: " + user_msg + "\nMODEL: "
        # Encode entire conversation
        context_ids = torch.tensor([tokenizer.encode(conversation)], dtype=torch.long, device=device)
        # We'll generate up to some tokens, but watch for stopping condition
        generated = model.generate(context_ids, max_new_tokens=300)[0]
        new_ids = generated[len(context_ids[0]):].tolist()
        raw_response = tokenizer.decode(new_ids)

        # We'll define a naive approach: stop at newline
        # If you have special tokens <|END_HERO|>, check for them in raw_response.
        stopping_tokens = ["\n", "<|END_HERO|>", "<|SEP|>"]
        final_response = raw_response
        for st in stopping_tokens:
            if st in final_response:
                final_response = final_response.split(st)[0]
        print("MODEL:", final_response.strip())

        # Add the model response to the conversation
        conversation += final_response.strip() + "\n"
>>>>>>> 7d5b632a9bd1c8ce37d6e30c350fbe1b062b9b2d

def main():
    if len(sys.argv) < 2:
        print("Usage: debate_gpt.py [train|eval|continuous|debate] [args...]")
        sys.exit(1)

    mode = sys.argv[1].strip().lower()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="", help="Input dataset file for training or reference.")
    parser.add_argument("--save", type=str, default="", help="Where to save the model (train mode).")
    parser.add_argument("--load", type=str, default="", help="Checkpoint to load (eval/continuous/debate).")
    parser.add_argument("--prompt", type=str, default="", help="Prompt text for eval/continuous.")
    parser.add_argument("--max-tokens", type=int, default=100, help="Number of tokens to generate in eval mode.")

<<<<<<< HEAD
    parser.add_argument("--context-size", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-layer", type=int, default=6)
=======
    parser.add_argument("--context-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=4)
>>>>>>> 7d5b632a9bd1c8ce37d6e30c350fbe1b062b9b2d
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--report", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args(sys.argv[2:])

    # Some required arguments if user didn't pass them
    # e.g. if user is training, we must have an input file
    if mode == "train":
        if not args.input:
            # Prompt the user
            user_in = input("No --input provided. Enter path to input file: ").strip()
            if not user_in:
                print("No input provided, exiting.")
                sys.exit(1)
            args.input = user_in

        if not args.save:
            user_save = input("No --save provided. Where to save model? (model.pth): ").strip()
            if not user_save:
                user_save = "model.pth"
            args.save = user_save

    elif mode in ["eval", "continuous", "debate"]:
        if not args.load:
<<<<<<< HEAD
            try:
                if mode in ["eval", "continuous"]:
                    args.load = "model/model_pretrained.pth"
                elif mode == "debate":
                    args.load = "model/model_finetuned.pth"
            except:
                print("Cannot load the model file from the default path.  "
                "Ensure that model/model_pretrained.pth(eval, continuous) or model/model_finetuned(debate)"
                "exist.  Alternatively, you can sepcify a different model file to load using the --load tag.  "
                "Use the -h flag for more information.")
                sys.exit(1)
=======
            user_load = input("No --load provided. Path to model checkpoint to load: ").strip()
            if not user_load:
                print("No checkpoint path, exiting.")
                sys.exit(1)
            args.load = user_load
>>>>>>> 7d5b632a9bd1c8ce37d6e30c350fbe1b062b9b2d

    # set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if mode not in ["train","eval","continuous","debate"]:
        print("Unknown mode:", mode)
        sys.exit(1)

<<<<<<< HEAD
    # If we have an input file, read it
=======
    # If we have an input file, let's read it
>>>>>>> 7d5b632a9bd1c8ce37d6e30c350fbe1b062b9b2d
    text_data = ""
    if args.input and os.path.exists(args.input):
        with open(args.input, "r", encoding="utf-8") as f:
            text_data = f.read()

    # Build tokenizer
<<<<<<< HEAD
    tokenizer = BPETokenizer()
=======
    tokenizer = CharacterTokenizer(text_data)
>>>>>>> 7d5b632a9bd1c8ce37d6e30c350fbe1b062b9b2d
    encoded_data = tokenizer.encode(text_data)

    # Build dataset
    dataset = TextDataset(
        encoded_data=encoded_data,
        context_size=args.context_size,
        batch_size=args.batch_size
    )

    # Build model
    model = GPTLanguageModel(
        vocab_size=len(tokenizer),
        n_embd=args.n_embd,
        context_size=args.context_size,
        n_head=args.n_head,
        n_layer=args.n_layer
    ).to(device)

    # Load checkpoint if needed
    if args.load and os.path.exists(args.load):
        print("Loading model from", args.load)
        model.load_state_dict(torch.load(args.load, map_location=device))

    if mode == "train":
        print("\n=== TRAIN MODE ===")
        train_loop(dataset, model, epochs=args.epochs, report=args.report, lr=args.lr)
        torch.save(model.state_dict(), args.save)
        print("Model saved to", args.save)

    elif mode == "eval":
        print("\n=== EVAL MODE ===")
        # We'll generate up to --max-tokens from the prompt
<<<<<<< HEAD
        model.eval()
=======
>>>>>>> 7d5b632a9bd1c8ce37d6e30c350fbe1b062b9b2d
        if args.prompt.strip():
            start_ids = tokenizer.encode(args.prompt)
            context_ids = torch.tensor([start_ids], dtype=torch.long, device=device)
        else:
            context_ids = torch.zeros((1,1), dtype=torch.long, device=device)

        out_ids = model.generate(context_ids, max_new_tokens=args.max_tokens)[0].tolist()
        result = tokenizer.decode(out_ids)
        print("Generated:\n", result)

    elif mode == "continuous":
        print("\n=== CONTINUOUS MODE ===")
        # We'll generate tokens indefinitely
        indefinite_continuous_generate(model, tokenizer, args.prompt)

    elif mode == "debate":
        print("\n=== DEBATE MODE ===")
        debate_mode(model, tokenizer)

if __name__ == "__main__":
<<<<<<< HEAD
    main()
=======
    main()
>>>>>>> 7d5b632a9bd1c8ce37d6e30c350fbe1b062b9b2d

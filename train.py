import argparse
import torch
from src.basic_model import BasicViT
from src.utils import train_with_params, load_and_transform

parser = argparse.ArgumentParser(add_help=True, description="Python script to run Basic ViT model training")
parser.add_argument('-f', metavar='f', type = float, help='fraction of the training set', default = 0.8)
parser.add_argument('-lr', metavar='lr', type = float, help='learning rate', default = 0.001)
parser.add_argument('-b_s', metavar='b_s', type = int , help='batch size', default = 32)
parser.add_argument('-e_n', metavar='e_n', type = int, help='number of training epochs', default = 1)
parser.add_argument('-u', metavar='u', type = bool , help='unfreeze weights for one epoch', default = False)
parser.add_argument('-a_b', metavar='a_b', type = bool, help='train with unfreezed weights during the first epoch', default = False)
parser.add_argument('-b', metavar='b', type = str , help='path to a file where base weights will be saved', default = "BASIC_MODEL.pt")
parser.add_argument('-t', metavar='t', type = str, help='path to a file where training results should be saved', default = "TRAINED_MODEL.pt")

if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datasets = load_and_transform(f = args.f)

    basic_model = BasicViT()
    torch.save(basic_model, args.b)

    params = {
        "lr": args.lr,
        "batch_size": args.b_s,
        "epochs_num": args.e_n,
        }
    specs = {
        "ViT_path": args.b,
        "unfreezed": args.u,
        "at_beginning": args.a_b
    }
    criterion = torch.nn.CrossEntropyLoss()

    accuracy, model = train_with_params(params,criterion, datasets, **specs)
    torch.save(model, args.t)
    torch.cuda.empty_cache()
    
    print(f"\nModel trained. Weights saved to the directory: {args.t}")
    print("Accuracy:", round(float(accuracy.detach()), 4))

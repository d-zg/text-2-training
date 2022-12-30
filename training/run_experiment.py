import argparse 
import train_methods
import synthetic_folder
import os 
from torch.utils.data import ConcatDataset



def main():
    parser = argparse.ArgumentParser(description ='script to train an efficientnet model on synthetic data')
    parser.add_argument('syntheticroot')
    parser.add_argument('datasetroot')
    parser.add_argument('minorityidx')
    parser.add_argument('idxdict')
    parser.add_argument('n')
    parser.add_argument('trimmed')
    parser.add_argument('syntheticsamples')
    parser.add_argument('epochs')
    parser.add_argument('checkpointfile')
    args = parser.parse_args()
    
    # num_synthetic = 100
    # synthetic_root = '~/stable-diffusion/outputs/txt2img-samples/apples'
    # dataset_root = '~/home/dzhang/efficientnet/data'
    # idx_dict = '/home/dzhang/home/text-2-training/training/idx.json' # idx dict for the untrimmed
    # num_classes = 6
    # epochs = 100
    # minority_class_idx = 0
    # amount_trimmed = 625
    # outputfile = 'applepietest'

    # ~/stable-diffusion/outputs/txt2img-samples/apple_pie/apple_pie/ ~/home/dzhang/efficientnet/data ~/home/text-2-training/idx_625_discarded_0.json 6 0 250 ~/home/text-2-training/outputfiles/checkpoints/apple_pie_control

    dataset = train_methods.get_food101(root=args.datasetroot)
    synthetic = train_methods.load_synthetic(args.syntheticroot, int(args.minorityidx))
    trimmedSynthetic = train_methods.random_trim(synthetic, int(args.syntheticsamples))
    idx_dict = train_methods.load_used_idxs(args.idxdict)
    train_val_idx = train_methods.get_shortened_idx(idx_dict, str(args.minorityidx), .25, .25, int(args.trimmed))
    datasets = train_methods.train_val_split_idx(dataset, train_val_idx)
    datasets['train'] = ConcatDataset(datasets=[datasets['train'], trimmedSynthetic])

    batch_size = 16
    dataloaders = train_methods.get_dataloaders(datasets=datasets, batch_size=batch_size)
    
    model = train_methods.fit_efficientnet_shape(int(args.n))
    criterion, optimizer = train_methods.make_criterion_optimizer(model)
    
    best_model, val_history = train_methods.train_model(model, dataloaders, criterion, optimizer, num_epochs=int(args.epochs))
    confusion = train_methods.eval_model(best_model, dataloaders, int(args.n))

    print('saving best model to ' + args.checkpointfile)
    checkpoint = {'state_dict' : best_model.state_dict(), 'optimizer' : optimizer.state_dict()}
    path = args.checkpointfile + ".pth.tar"
    train_methods.save_checkpoint(state=checkpoint, filename=path)
    train_methods.save_train_history(args.checkpointfile, val_history)
    train_methods.save_test_results(args.checkpointfile, confusion)
    
if __name__ == "__main__":
    main()
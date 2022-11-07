import train_methods
import argparse 
import torchvision

def main():
    parser = argparse.ArgumentParser(description ='script to train an efficientnet model on synthetic data')
    parser.add_argument('syntheticroot')
    parser.add_argument('datasetroot')
    parser.add_argument('idxdict')
    parser.add_argument('n')
    args = parser.parse_args()

    dataset = train_methods.get_food101(root=args.datasetroot)
    split_idx = train_methods.load_used_idxs(root=args.idxdict) 

    datasets = train_methods.train_val_split_idx(dataset, split_idx)
    batch_size = 32
    dataloaders = train_methods.get_dataloaders(datasets=datasets, batch_size=batch_size)
    model = train_methods.fit_efficientnet_shape(args.n)
    criterion, optimizer = train_methods.make_criterion_optimizer(model)
    best_model = train_methods.train_model(model, dataloaders, criterion, optimizer, num_epochs=100)
    
if __name__ == "__main__":
    main()
import train_methods
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description ='script to turn an idx dict separated by class into a train/val/test idx dict')
    parser.add_argument('idxdict')
    parser.add_argument('classId')
    parser.add_argument('discard')
    parser.add_argument('outputfile')
    args = parser.parse_args()
    
    idx_dict = train_methods.load_used_idxs(args.idxdict)
    # print(idx_dict)
    train_val_idx = train_methods.get_shortened_idx(idx_dict, str(args.classId), .25, .25, int(args.discard))

    with open(args.outputfile, "w") as outfile:
        print('writing train/val/test partition dict to json file ' + args.outputfile)
        json.dump(train_val_idx, outfile)

if __name__ == "__main__":
    main()
    
import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann', type=str, default='datasets/ILSVRC2012/ILSVRC2012_img_val.json')
    parser.add_argument('--f', type=str, default='tools/ilsvrc2012_classes_name.txt')
    parser.add_argument('--output',type=str, default='datasets/ILSVRC2012/ILSVRC2012_img_val_converted.json')
    args = parser.parse_args()
    with open(args.f,'r') as f:
        lines = f.readlines()
        d = {}
        for line in lines:
            k,v = line.split(':')
            d[k.strip()] = v.split(',')[0].strip()
    
    print('Loading', args.ann)
    data = json.load(open(args.ann, 'r'))
    data['categories'] = [{'id':cat['id'],'name':d[cat['name']]} for cat in data['categories']]
    print('Saving to', args.output)
    json.dump(data, open(args.output, 'w'))
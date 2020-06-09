from utils import *
import random

class MathExprDataset(Dataset):
    def __init__(self, split='train', numSamples=None, randomSeed=None):
        super(MathExprDataset, self).__init__()
        
        self.split = split
        self.dataset = json.load(open('./data/expr_%s.json'%split))
        if numSamples:
            if randomSeed:
                random.seed(randomSeed)
                random.shuffle(self.dataset)
            self.dataset = self.dataset[:numSamples]
            
        for x in self.dataset:
            x['len'] = len(x['expr'])

        self.img_transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (1,))])
    
    def __getitem__(self, index):
        sample = deepcopy(self.dataset[index])
        img_seq = []
        for img_path in sample['img_paths']:
            img = Image.open(img_dir+img_path).convert('L')
            #print(img.size, img.mode)
            img = self.img_transform(img)
            img_seq.append(img)
        del sample['img_paths']
        
        label_seq = [sym2id(sym) for sym in sample['expr']]
        sample['img_seq'] = img_seq
        sample['label_seq'] = label_seq
        sample['len'] = len(sample['expr'])

        res = eval(sample['expr'])
        res = round(res, res_precision)
        sample['res'] = res
        return sample
            
    
    def __len__(self):
        return len(self.dataset)

    def filter_by_len(self, max_len):
        self.dataset = [x for x in self.dataset if x['len'] <= max_len]


def MathExpr_collate(batch):
    max_len = np.max([x['len'] for x in batch])
    zero_img = torch.zeros_like(batch[0]['img_seq'][0])
    for sample in batch:
        sample['img_seq'] += [zero_img] * (max_len - sample['len'])
        sample['img_seq'] = torch.stack(sample['img_seq'])
        
        sample['label_seq'] += [sym2id('UNK')] * (max_len - sample['len'])
        sample['label_seq'] = torch.tensor(sample['label_seq'])
        
    batch = default_collate(batch)
    return batch
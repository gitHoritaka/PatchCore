from model import PatchCore
from data import MvTecDataset,MVTecTrainDataset
from torch.utils.data import DataLoader

def train():
    train_dataset = MvTecDataset("datasets/train/good")
    train_dataloader = DataLoader(train_dataset,batch_size=1)
    model = PatchCore()
    model.fit(train_dataloader)
def test():
    train_dataset = MvTecDataset("datasets/train/good")
    train_dataloader = DataLoader(train_dataset,batch_size=1)
    test_dataset = MvTecDataset("datasets/train/not-good")
    test_dataloader = DataLoader(test_dataset)
    model = PatchCore()
    cnt = 0
    for img_batch,label in train_dataloader:
        cnt +=1
        if cnt > 5:
            break
        s,smap = model.predict(img_batch)
        print(s,label)
    cnt = 0
    for img_batch,label in test_dataloader:
        cnt += 1
        if cnt > 5:
            break
        s,smap = model.predict(img_batch)
        print(s,label)
test()

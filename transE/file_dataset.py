from torch.utils.data import Dataset
class TransEDataset(Dataset):
    def __init__(self, datapath) -> None:
        super(TransEDataset).__init__()
        with open(datapath, "r") as fp:
            self.data = [line.strip().split("\t") for line in fp]   # head, relation(click or follow), tail
        self.data_length = len(self.data)
    def __len__(self): 
        return self.data_length
    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1], self.data[index][2]    # head, relation(click or follow), tail
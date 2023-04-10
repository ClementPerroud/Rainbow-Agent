import numpy as np
class SumTree:
    def __init__(self, size):
        self.size = size
        
        self.node_layers = [np.array([0.]*size)]
        layer_size = size
        while layer_size > 1:
            layer_size = (layer_size + 1)//2
            self.node_layers.append(np.array([0.]*layer_size))
        self.layer_len = len(self.node_layers)
        self.layer_lens = [len(layer) for layer in self.node_layers]
        self.new_leafs = []
        self.count = 0
    def __setitem__(self, loc, value):
        if isinstance(loc, int):
            idx = loc
            change = value - self.node_layers[0][idx]
            self.node_layers[0][idx] = value

            for idx_layer in range(1, self.layer_len):
                idx = idx // 2 # Get parent index
                self.node_layers[idx_layer][idx] += change
        else: # List, array ...
            for i in range(len(loc)): self.__setitem__(loc[i], value[i])
    def __getitem__(self, loc):
        return self.node_layers[0][loc]
    def add(self, i):
        self[i] = 3 * self.sum() / self.count if self.count > 0 else 1
        self.new_leafs.append(i)
        self.count = min(self.count + 1, self.sum())


    def sum(self):
        return self.node_layers[self.layer_len -1][0]
    def sample(self, batch_size):
        batch = []
        while len(self.new_leafs) > 0 and len(batch) < batch_size:
            batch.append(self.new_leafs.pop(0))
        batch_size -= len(batch)
        random_cumsums = np.random.rand(batch_size)*self.sum()
        for i in range(batch_size):
            cumsum = random_cumsums[i]

            idx = 0
            for i_layer in range(self.layer_len-1, 0, -1):
                left_child_index, right_child_index = idx*2, idx*2+1
                if cumsum < self.node_layers[i_layer-1][left_child_index]:
                    idx = left_child_index
                else:
                    idx = right_child_index
                    cumsum -= self.node_layers[i_layer-1][left_child_index]
            batch.append(idx)
        return batch

import numpy as np

class enumerate_all_labels(object):
    def __init__(self):

        label_dim = 24
        n_item = 3

        N = 0
        for i in range(n_item):
            S = 1
            for j in range(i+1):
                S = S * ( label_dim - j )/( j+1 )
            N += S
        N = int(N)

        labels = np.zeros([N, label_dim], dtype=int)

        idx = 0
        for d in range(label_dim):            
            labels[idx][d] = 1
            idx += 1

        for d1 in range(label_dim):
            for d2 in range(label_dim):
                if d2>d1:                    
                    labels[idx][[d1,d2]] = 1
                    idx += 1

        for d1 in range(label_dim):
            for d2 in range(label_dim):
                for d3 in range(label_dim):
                    if d2>d1 and d3 >d2:                        
                        labels[idx][[d1,d2,d3]] = 1
                        idx += 1

        self.labels = labels
        self.N = N

    def get(self, n_label):
        idx = np.random.randint(0, self.N, n_label)
        return self.labels[idx]

if __name__ == "__main__":
    listed_labels = enumerate_all_labels()
    labels = listed_labels.get(10)
    print(labels)
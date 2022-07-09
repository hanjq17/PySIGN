from torch_geometric.data import Data


def from_pyg(attrs):
    def transform(pyg_data):
        info = {}
        for element in attrs:
            if element == 'x':
                our_name = 'h'
            elif element == 'pos':
                our_name = 'x'
            elif element == 'z':
                our_name = 'charge'
            else:
                our_name = element
            info[our_name] = pyg_data[element]
        return Data(**info)

    return transform

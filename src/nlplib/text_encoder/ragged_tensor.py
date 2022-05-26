import numpy as np

from collections.abc import Iterable

def padded_tensor_shape(ragged_tensor: Iterable) -> np.ndarray:
    """ Returns the shape of the padded tensor given <ragged_tensor>.
    Args:
        ragged_tensor (Iterable): An unevenly nested iterable.
    Returns:
        shape (np.ndarray): The shape of the fully padded tensor.
    """
    if isinstance(ragged_tensor, str) or not isinstance(ragged_tensor, Iterable):
        return np.array([], dtype=int)

    subtensor_shapes = [
        padded_tensor_shape(ragged_subtensor)
        for ragged_subtensor in ragged_tensor
    ]

    max_dim = max([ shape.size for shape in subtensor_shapes ])
    current_dim_size = len(subtensor_shapes)

    tensor_shape = np.stack([
        np.append(subtensor_shape, [0] * (max_dim - subtensor_shape.size))
        for subtensor_shape in subtensor_shapes
    ]).max(axis=0)

    return np.append(
        np.array([ current_dim_size ]),
        tensor_shape
    ).astype(int)

def padded_tensor(ragged_tensor: Iterable, pad_value: any, dtype = None,
    tensor_shape: np.ndarray = None) -> np.ndarray:
    """ Pads <ragged_tensor> with <pad_value>.
    Args:
        ragged_tensor (Iterable): An unevenly nested iterable.
        pad_value (any): The value to pad the tensor with. Must be compatible
                with the exisiting values of <ragged_batch> and <dtype>.
        dtype (type): The data type to cast the padded tensor to.
        tensor_shape (np.ndarray): The expected shape of the fully padded tensor.
                When un-specified, calls <padded_tensor_shape> method to
                determine the appropriate container for the tensor.
    Returns:
        shape (np.ndarray): The shape of the fully padded tensor.
    """
    if tensor_shape is None: # Base dimension
        tensor_shape = padded_tensor_shape(ragged_tensor)

    if isinstance(ragged_tensor, str) or not isinstance(ragged_tensor, Iterable):
        if not tensor_shape.size: # Last dimension
            return ragged_tensor
        
        padded_ragged_batch = np.full(shape=tensor_shape,
                fill_value=pad_value, dtype=dtype)
        
        padded_ragged_batch[tuple([0] * tensor_shape.size)] = ragged_tensor
        return padded_ragged_batch
    
    current_dim_size = ragged_tensor.shape[0] \
            if isinstance(ragged_tensor, np.ndarray) else \
            len(ragged_tensor)

    return np.concatenate([
        np.stack([
            padded_tensor(ragged_subtensor, pad_value=pad_value,
                    tensor_shape=tensor_shape[1:])
            for ragged_subtensor in ragged_tensor
        ], axis=0),
        np.full(
            shape=[ tensor_shape[0] - current_dim_size, *tensor_shape[1:] ],
            fill_value=pad_value,
            dtype=dtype
        )
    ])

if __name__ == "__main__":
    pass

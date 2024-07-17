import itertools

def get_batch_direct(dataloader, i):
    """
    Get the batch of index i of the dataloader
    """
    return next(itertools.islice(dataloader, i, None))
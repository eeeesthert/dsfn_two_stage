"""Optional TensorBoard writer without coupling model code to logging."""
def create_writer(path):
 from torch.utils.tensorboard import SummaryWriter
 return SummaryWriter(path)

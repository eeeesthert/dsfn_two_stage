"""TensorBoard image visualization helpers."""
def log_images(writer,images,step):
    for name,value in images.items(): writer.add_images(name,value.detach().cpu().clamp(-1,1)*.5+.5,step)

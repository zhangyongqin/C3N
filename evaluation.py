import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image

from util.image import unnormalize
import time
def evaluate(model, dataset, device, filename):
    image, mask, gt = zip(*[dataset[i] for i in range(8)])
    image = torch.stack(image)
    mask = torch.stack(mask)
    gt = torch.stack(gt)
    with torch.no_grad():
        time_start= time.time()
        output, _ = model(image.to(device), mask.to(device))
        time_end = time.time()
        print('time cost', time_end - time_start, 's')
    output = output.to(torch.device('cpu'))
    output_comp = mask * image + (1 - mask) * output

    grid = make_grid(
        torch.cat((unnormalize(image), mask, unnormalize(output),
                   unnormalize(output_comp), unnormalize(gt)), dim=0))
    save_image(grid, filename)

from seed import seed_all
from model import SHOPEE_HIRE_ME_MODEL
import torch

if __name__ == "__main__":
    seed_all()
    device = "cuda"
    model = SHOPEE_HIRE_ME_MODEL(
        num_classes=2,
        dropout=0.0,
        embedding_size=8,
        backbone="vgg16",
        pretrained=True,
    )
    model.to(device)

    # GOOD Practice to test forward pass, print out sin's inputs shape
    # torch.Size([16, 3, 512, 512]) torch.Size([16])
    x = torch.rand(2, 3, 32, 32).to(
        device
    )  # 16 batch size, of 512 size images of 3 channels
    y = torch.rand(2).to(device)
    # with torch.no_grad():
    y1 = model(x, y)
    print("[forward test]")
    print("input:\t{}\noutput:\t{}".format(x.shape, y1.shape))
    print("output value", y1[0][0])
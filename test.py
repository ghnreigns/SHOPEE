from seed import seed_all
from model import *
import torch
from config import CONFIG

if __name__ == "__main__":
    seed_all()
    device = "cuda"

    torch.cuda.empty_cache()
    model = TEST_MODEL(
        channel_size=512,
        out_feature=11014,
        dropout=0.0,
        backbone="densenet121",
        pretrained=True,
    )
    model = model.to(device)

    # GOOD Practice to test forward pass, print out sin's inputs shape
    # torch.Size([16, 3, 512, 512]) torch.Size([16])
    # x = torch.rand(2, 3, 32, 32).to(
    #     device
    # )  # 16 batch size, of 512 size images of 3 channels
    # y = torch.rand(2).to(device)
    # # with torch.no_grad():
    # y1 = model(x, y)
    # print("[forward test]")
    # print("input:\t{}\noutput:\t{}".format(x.shape, y1.shape))
    # print("output value", y1[0][0])

    x = torch.rand(2, 3, 512, 512).to(
        device
    )  # 16 batch size, of 512 size images of 3 channels
    y = torch.rand(2).to(device)
    # with torch.no_grad():
    y1 = model(x, y)
    print("[forward test]")
    print("input:\t{}\noutput:\t{}".format(x.shape, y1.shape))
    print("output value", y1[0][0])
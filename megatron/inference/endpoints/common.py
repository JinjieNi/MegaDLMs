# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import threading

import torch

GENERATE_NUM = 0
BEAM_NUM = 1
LOCK = threading.Lock()


def send_do_generate():
    choice = torch.tensor([GENERATE_NUM], dtype=torch.long, device="cuda")
    torch.distributed.broadcast(choice, 0)


def send_do_beam_search():
    choice = torch.tensor([BEAM_NUM], dtype=torch.long, device="cuda")
    torch.distributed.broadcast(choice, 0)

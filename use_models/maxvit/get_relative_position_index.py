import torch


def get_relative_position_index(M=2):
    # 形成相对位置坐标
    a = torch.arange(M)
    b = torch.arange(M)
    x, y = torch.meshgrid([a, b])
    print("x\n", x)
    print("y\n", y)
    # stack([M, M], [M, M]) -> [2, M, M]
    coords = torch.stack((x, y))
    print("coords\n", coords)
    # [2, M, M] -> flatten(1) -> [2, MM]
    coords_flatten = torch.flatten(coords, 1)
    print("coords_flatten\n", coords_flatten)
    # [2, MM, 1] - [2, 1, MM] -> [2, MM, MM]
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    print("coords_flatten[:, :, None]\n", coords_flatten[:, :, None])
    print("coords_flatten[:, None, :]\n", coords_flatten[:, None, :])
    print("relative_coords\n", relative_coords)
    # [2, MM, MM] -> [MM, MM, 2]    # 每个 [MM, 2] 都是一组相对位置坐标，这一定是魔法。。。
    relative_coords = relative_coords.permute(1, 2, 0)
    print("relative_coords\n", relative_coords)
    
    # 行和列标全部加上 M - 1，这里 M - 1 = 1
    relative_coords[...] += M - 1
    print("relative_coords\n", relative_coords)
    # 行标乘上 2M - 1，这里 2M - 1 = 3
    relative_coords[:, :, 0] *= 2 * M - 1
    print("relative_coords\n", relative_coords)
    # 行标和列表相加
    # [MM, MM, 2] -> sum(-1) -> [MM, MM]，其中每个值代表一种相对位置坐标，共有 9 种坐标（M=2时），则值范围为 0~8
    relative_coords  = torch.sum(relative_coords, dim=-1)
    print("relative_coords\n", relative_coords)
    
    return relative_coords
    
    
if __name__ == "__main__":
    num_heads = 8
    M = 2
    n = M * M
    relative_coords = get_relative_position_index(M)
    # [(2M-1)*(2M-1), h]
    relative_position_bias_table = torch.zeros(((2*M-1)**2, num_heads))
    # [MM*MM, h] = [nn, h] -> reshape -> [n, n, -1] = [n, n, h]
    relative_position_bias = relative_position_bias_table[relative_coords.view(-1)].reshape(n, n, -1)
    # [n, n, h] -> permute -> [h, n, n] -> unsqueeze -> [1, h, n, n]
    relative_position_bias = relative_position_bias.permute(2, 0, 1).unsqueeze(0)
    print(relative_position_bias.shape)